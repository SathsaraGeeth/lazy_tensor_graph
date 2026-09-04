"""validation and  from AST to operation IR"""

from dataclasses import dataclass
import re

from ast_nodes import Operation, ValueDecl
from parser import DTYPE

DEVICES = {"generic", "cpu", "cuda"}
IDENT = re.compile(r"[A-Za-z_]\w*")


@dataclass(frozen=True)
class Value:
    kind: str
    name: str
    type_key: str
    dtypes: tuple[str, ...]
    shape: tuple[str, ...]
    count: int


@dataclass
class ValidatedOperation:
    ast: Operation
    values: list[Value]
    targets: dict[str, tuple[str, ...]]

    @property
    def inputs(self): return [v for v in self.values if v.kind == "in"]
    @property
    def outputs(self): return [v for v in self.values if v.kind == "out"]
    @property
    def parameters(self): return [v for v in self.values if v.kind == "param"]


def validate(operation: Operation) -> ValidatedOperation:
    error = lambda message: _error(operation, message)
    if not operation.types: error("at least one type definition is required")
    if not operation.algorithms: error("at least one algorithm is required")
    if operation.hints.pointwise is None: error("pointwise hint is required")
    if not 0 <= operation.id <= 0x0fffffff: error("id must fit in 28 bits")

    names = set()
    for group in (operation.types, operation.devices, operation.dimensions,
                  operation.shapes):
        for name in group:
            if name in names: error(f"duplicate definition '{name}'")
            names.add(name)
    for definition in operation.types.values():
        if not definition.dtypes or len(set(definition.dtypes)) != len(definition.dtypes):
            error(f"type '{definition.name}' must be a non-empty set without duplicates")
        if any(dtype not in DTYPE for dtype in definition.dtypes):
            error(f"type '{definition.name}' contains an unsupported dtype")
    for definition in operation.devices.values():
        if len(set(definition.devices)) != len(definition.devices):
            error(f"devices '{definition.name}' must not contain duplicates")
        if any(device not in DEVICES for device in definition.devices):
            error(f"devices '{definition.name}' contains an unsupported device")
    for dimension in operation.dimensions.values():
        if dimension.value is not None and dimension.value <= 0:
            error(f"dimension '{dimension.name}' must be positive")
    for shape in operation.shapes.values():
        _validate_shape(operation, shape.dimensions)

    values, value_names = [], set()
    for value in operation.values:
        if value.name in names or value.name in value_names:
            error(f"duplicate name '{value.name}'")
        value_names.add(value.name)
        values.append(_resolve_value(operation, value))
    outputs = [value for value in values if value.kind == "out"]
    if not outputs: error("at least one output is required")

    algorithms = [algorithm.name for algorithm in operation.algorithms]
    if len(set(algorithms)) != len(algorithms): error("algorithm names must be unique")
    if set(operation.hints.targets) != set(algorithms):
        error("every algorithm must have exactly one target")
    if set(operation.hints.operations) != set(algorithms):
        error("every algorithm must have exactly one operations hint")
    for name in (*operation.hints.requirements, *(x[0] for x in operation.hints.preferences)):
        if name not in algorithms: error(f"hint references unknown algorithm '{name}'")
    targets = {}
    for name, reference in operation.hints.targets.items():
        if reference in DEVICES: targets[name] = (reference,)
        elif reference in operation.devices: targets[name] = operation.devices[reference].devices
        else: error(f"unknown device target '{reference}'")
    for name, expression in operation.hints.requirements.items():
        _validate_condition(operation, name, expression)
    for name, expression in operation.hints.preferences:
        if expression is not None: _validate_condition(operation, name, expression)
    for name, expression in operation.hints.operations.items():
        _validate_cost(operation, name, expression)
    for device in DEVICES:
        if not any(device in targets[name] and name not in operation.hints.requirements
                   for name in algorithms):
            error(f"device '{device}' has no unconditional fallback algorithm")
    if operation.hints.pointwise and any(value.shape != outputs[0].shape for value in outputs):
        error("all pointwise outputs must have the same shape")
    _validate_dispatch_types(operation, values)
    return ValidatedOperation(operation, values, targets)


def validate_tree(operations: list[ValidatedOperation]) -> None:
    ids, names = set(), set()
    for operation in operations:
        if operation.ast.id in ids: _error(operation.ast, f"duplicate id 0x{operation.ast.id:x}")
        if operation.ast.name in names: _error(operation.ast, f"duplicate name '{operation.ast.name}'")
        ids.add(operation.ast.id); names.add(operation.ast.name)


def _resolve_value(operation: Operation, value: ValueDecl) -> Value:
    if value.kind == "param":
        if value.type_name not in DTYPE: _error(operation, f"parameter '{value.name}' needs a concrete dtype")
        return Value(value.kind, value.name, value.type_name, (value.type_name,), (), value.count)
    if value.type_name in DTYPE:
        dtypes, key = (value.type_name,), value.type_name
    elif value.type_name in operation.types:
        dtypes, key = operation.types[value.type_name].dtypes, value.type_name
    else:
        _error(operation, f"unknown type '{value.type_name}'")
    shape = operation.shapes[value.shape].dimensions if isinstance(value.shape, str) and value.shape in operation.shapes else value.shape
    if not isinstance(shape, tuple): _error(operation, f"unknown shape '{value.shape}'")
    _validate_shape(operation, shape)
    return Value(value.kind, value.name, key, dtypes, shape, 1)


def _validate_shape(operation: Operation, shape: tuple[str, ...]) -> None:
    for dimension in shape:
        if dimension in ("*", "**") or dimension.isdigit(): continue
        if not IDENT.fullmatch(dimension) or dimension not in operation.dimensions:
            _error(operation, f"unknown dimension '{dimension}'")


def _validate_dispatch_types(operation: Operation, values: list[Value]) -> None:
    inputs = [v for v in values if v.kind == "in"]
    outputs = [v for v in values if v.kind == "out"]
    selectable = {outputs[0].type_key}
    if inputs: selectable.add(inputs[0].type_key)
    for value in inputs + outputs:
        if len(value.dtypes) > 1 and value.type_key not in selectable:
            _error(operation, f"type of '{value.name}' cannot be selected from output/first-input signature")


def _validate_condition(operation: Operation, name: str, expression: str) -> None:
    if re.search(r"\+\+|--|(?<![=!<>])=(?!=)|\b[A-Za-z_]\w*\s*\(", expression):
        _error(operation, f"condition for '{name}' has a forbidden side effect or call")
    if ";" in expression or "{" in expression or "}" in expression:
        _error(operation, f"condition for '{name}' is not a scalar expression")


def _validate_cost(operation: Operation, name: str, expression: str) -> None:
    _validate_condition(operation, name, expression)
    if not expression.strip():
        _error(operation, f"operations hint for '{name}' is empty")
    tensor_names = {
        f"{value.name}_{suffix}"
        for value in operation.values if value.kind in ("in", "out")
        for suffix in ("size", "rank")
    }
    allowed = set(operation.dimensions) | {"point_count"} | tensor_names
    for token in re.findall(r"\b[A-Za-z_]\w*\b", expression):
        if token not in allowed:
            _error(operation,
                   f"operations hint for '{name}' uses unknown name '{token}'")
    if re.search(r"[^A-Za-z0-9_\s()+*/%<>&|^~?:.-]", expression):
        _error(operation,
               f"operations hint for '{name}' is not an integer expression")


def _error(operation: Operation, message: str):
    raise ValueError(f"{operation.source}: {message}")
