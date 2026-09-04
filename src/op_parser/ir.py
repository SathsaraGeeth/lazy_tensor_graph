"""Lower validated operations to one C translation unit and LLVM bitcode."""

import itertools
import pathlib
import subprocess
import tempfile

from validator import ValidatedOperation, Value

CTYPE = {"int8":"int8","int16":"int16","int32":"int32","int64":"int64",
         "uint8":"uint8","uint16":"uint16","uint32":"uint32","uint64":"uint64",
         "real32":"real32","real64":"real64"}
ENUM = {name: index for index, name in enumerate(CTYPE)}
SIZE = {"int8":1,"uint8":1,"int16":2,"uint16":2,"int32":4,"uint32":4,
        "real32":4,"int64":8,"uint64":8,"real64":8}
ALIGN = SIZE


def _specialization_symbol(operation, output_dtype, input_dtype):
    """Return the canonical C/LLVM symbol for one dtype specialization."""
    return (
        f"tensor_op_{operation.ast.id:08x}_"
        f"{ENUM[output_dtype]}_{ENUM[input_dtype]}"
    )


def emit_bitcode(operations: list[ValidatedOperation], output: pathlib.Path,
                 device: str) -> list[tuple[ValidatedOperation, int, int, str]]:
    records, functions = [], []
    for operation in operations:
        for assignment in _assignments(operation):
            output_dtype = assignment[operation.outputs[0].type_key]
            input_dtype = assignment[operation.inputs[0].type_key] if operation.inputs else output_dtype
            symbol = _specialization_symbol(
                operation, output_dtype, input_dtype)
            functions.append(_function(operation, assignment, device, symbol))
            records.append((operation, ENUM[output_dtype], ENUM[input_dtype], symbol))
    source = '#include "tensor_core.h"\n#include "dtype.h"\n#include "device.h"\n\n' + "\n".join(functions)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tensor-op-") as directory:
        source_path = pathlib.Path(directory) / "operations.c"
        source_path.write_text(source, encoding="utf-8")
        result = subprocess.run(
            ["clang-18", "-std=c11", "-O1", "-emit-llvm", "-c", "-I",
             str(pathlib.Path(__file__).resolve().parents[2] / "include"),
             str(source_path), "-o", str(output)], capture_output=True, text=True)
        if result.returncode:
            raise ValueError(f"generated C compilation failed:\n{result.stderr}")
    return records


def _assignments(operation):
    keys, choices = [], []
    for value in operation.inputs + operation.outputs:
        if value.type_key not in keys:
            keys.append(value.type_key); choices.append(value.dtypes)
    for values in itertools.product(*choices):
        yield dict(zip(keys, values))


def _function(operation, assignment, device, symbol):
    algorithms = [a for a in operation.ast.algorithms if device in operation.targets[a.name]]
    if not algorithms:
        raise ValueError(f"{operation.ast.source}: no algorithm targets '{device}'")
    declarations, dimensions = _bindings(operation, assignment)
    helpers = "\n".join(_helper(operation, assignment, algorithm, declarations,
                               f"{symbol}_{algorithm.name}")
                        for algorithm in algorithms)
    calls = _selection(operation, algorithms, symbol)
    point_dispatch = (
        _point_selection(operation, algorithms, symbol)
        if operation.ast.hints.pointwise else "")
    dims = ", ".join(dimensions) or "0"
    inputs = ", ".join(f"input_tensors[{i}]->data->ptr" for i in range(len(operation.inputs))) or "NULL"
    condition_bindings = _condition_bindings(operation)
    parameter_bytes = _parameter_bytes(operation)
    return f"""{helpers}
boolean {symbol}(
    tensor *output, const tensor *const *input_tensors, extent input_count,
    const void *parameters, extent parameter_bytes) {{
  if (input_count != {len(operation.inputs)} ||
      parameter_bytes != {parameter_bytes} ||
      ({parameter_bytes} && !parameters)) return true;
  void *outputs[] = {{ output->data->ptr }};
  const void *inputs[] = {{ {inputs} }};
  extent dimensions[] = {{ {dims} }};
{chr(10).join(condition_bindings)}
{calls}
}}
{point_dispatch}
"""


def _bindings(operation, assignment):
    declarations, offset = [], 0
    for name in operation.ast.types:
        if name in assignment:
            declarations.append(f"  typedef {CTYPE[assignment[name]]} {name};")
    pointwise, output_shape = operation.ast.hints.pointwise, operation.outputs[0].shape
    for index, value in enumerate(operation.inputs):
        ctype = CTYPE[assignment[value.type_key]]
        if pointwise and value.shape == output_shape:
            declarations.append(f"  {ctype} {value.name} = ((const {ctype} *)inputs[{index}])[point_index];")
        else:
            declarations.append(f"  const {ctype} *restrict {value.name} = (const {ctype} *)inputs[{index}];")
        declarations += _type_facts(value.name, assignment[value.type_key])
    for index, value in enumerate(operation.outputs):
        ctype = CTYPE[assignment[value.type_key]]
        if pointwise:
            declarations.append(f"  {ctype} {value.name} = (({ctype} *)outputs[{index}])[point_index];")
        else:
            declarations.append(f"  {ctype} *restrict {value.name} = ({ctype} *)outputs[{index}];")
        declarations += _type_facts(value.name, assignment[value.type_key])
    for value in operation.parameters:
        size, align = SIZE[value.type_key], ALIGN[value.type_key]
        offset = (offset + align - 1) // align * align
        ctype, suffix = CTYPE[value.type_key], f"[{value.count}]" if value.count > 1 else ""
        if value.count > 1:
            declarations.append(f"  const {ctype} *restrict {value.name} = (const {ctype} *)((const uint8 *)parameters + {offset});")
        else:
            declarations.append(f"  {ctype} {value.name} = *(const {ctype} *)((const uint8 *)parameters + {offset});")
        offset += size * value.count
    tensors = operation.inputs + operation.outputs
    symbols = []
    for value in tensors:
        for dim in value.shape:
            if dim not in ("*", "**") and not dim.isdigit() and dim not in symbols: symbols.append(dim)
    dimensions = []
    for symbol in symbols:
        expression = None
        fixed = operation.ast.dimensions[symbol].value
        if fixed is not None: expression = str(fixed)
        else:
            for index, value in enumerate(operation.inputs):
                if symbol in value.shape:
                    expression = f"((const extent *)input_tensors[{index}]->shape)[{value.shape.index(symbol)}]"; break
            if expression is None:
                for value in operation.outputs:
                    if symbol in value.shape:
                        expression = f"((const extent *)output->shape)[{value.shape.index(symbol)}]"; break
        dimensions.append(expression)
        declarations.append(f"  extent {symbol} = {expression};")
    for index, value in enumerate(operation.inputs):
        declarations += [f"  extent {value.name}_size = input_tensors[{index}]->size;",
                         f"  extent {value.name}_rank = input_tensors[{index}]->rank;",
                         f"  const extent *{value.name}_shape = (const extent *)input_tensors[{index}]->shape;"]
    for value in operation.outputs:
        declarations += [f"  extent {value.name}_size = output->size;",
                         f"  extent {value.name}_rank = output->rank;",
                         f"  const extent *{value.name}_shape = (const extent *)output->shape;"]
    declarations.append("  extent point_count = output->size;")
    return declarations, dimensions


def _type_facts(name, dtype):
    lower = "DTYPE_REAL32_MIN" if dtype == "real32" else "DTYPE_REAL64_MIN" if dtype == "real64" else "0" if dtype.startswith("uint") else f"DTYPE_{dtype.upper()}_MIN"
    upper = f"DTYPE_{dtype.upper()}_MAX"
    return [f"  const real64 {name}_min = (real64)({lower});",
            f"  const real64 {name}_max = (real64)({upper});",
            f"  const real64 {name}_range = {name}_max - {name}_min;",
            f"  const boolean {name}_float = {'true' if dtype.startswith('real') else 'false'};"]


def _condition_bindings(operation):
    declarations, offset = [], 0
    for value in operation.parameters:
        size, align = SIZE[value.type_key], ALIGN[value.type_key]
        offset = (offset + align - 1) // align * align
        ctype = CTYPE[value.type_key]
        if value.count > 1:
            declarations.append(f"  const {ctype} *restrict {value.name} = (const {ctype} *)((const uint8 *)parameters + {offset});")
        else:
            declarations.append(f"  {ctype} {value.name} = *(const {ctype} *)((const uint8 *)parameters + {offset});")
        offset += size * value.count
    for symbol, definition in operation.ast.dimensions.items():
        if definition.value is not None:
            declarations.append(f"  extent {symbol} = {definition.value};")
            continue
        expression = None
        for index, value in enumerate(operation.inputs):
            if symbol in value.shape:
                expression = f"((const extent *)input_tensors[{index}]->shape)[{value.shape.index(symbol)}]"; break
        if expression is None:
            for value in operation.outputs:
                if symbol in value.shape:
                    expression = f"((const extent *)output->shape)[{value.shape.index(symbol)}]"; break
        if expression:
            declarations.append(f"  extent {symbol} = {expression};")
    declarations.append("  extent point_count = output->size;")
    return declarations


def _parameter_bytes(operation):
    offset, maximum_alignment = 0, 1
    for value in operation.parameters:
        alignment = ALIGN[value.type_key]
        maximum_alignment = max(maximum_alignment, alignment)
        offset = (offset + alignment - 1) // alignment * alignment
        offset += SIZE[value.type_key] * value.count
    return (offset + maximum_alignment - 1) // maximum_alignment * maximum_alignment


def _helper(operation, assignment, algorithm, declarations, helper_name):
    stores = "\n".join(
        f"  (({CTYPE[assignment[v.type_key]]} *)outputs[{i}])[point_index] = {v.name};"
        for i, v in enumerate(operation.outputs)) if operation.ast.hints.pointwise else ""
    body = algorithm.body.replace("parallel for", "for")
    if stores:
        body = body.replace("return false;", stores + "\n  return false;")
    return f"""static boolean {helper_name}(
    tensor *output, const tensor *const *input_tensors,
    void *const *outputs, const void *const *inputs,
    const void *parameters, const extent *dimensions, extent point_index) {{
  (void)dimensions;
{chr(10).join(declarations)}
{body}
}}
"""


def _selection(operation, algorithms, symbol):
    lines = []
    eligible = {a.name: operation.ast.hints.requirements.get(a.name, "true") for a in algorithms}
    ordered = []
    for name, condition in operation.ast.hints.preferences:
        if name in eligible and name not in ordered:
            ordered.append(name)
            eligible[name] = f"({eligible[name]}) && ({condition or 'true'})"
    ordered += [a.name for a in algorithms if a.name not in ordered]
    for name in ordered:
        if operation.ast.hints.pointwise:
            call = (f"if ({eligible[name]}) {{\n"
                    f"    for (extent point_index = 0; point_index < output->size; ++point_index)\n"
                    f"      if ({symbol}_{name}(output, input_tensors, outputs, inputs, parameters, dimensions, point_index)) return true;\n"
                    f"    return false;\n  }}")
        else:
            call = f"if ({eligible[name]}) return {symbol}_{name}(output, input_tensors, outputs, inputs, parameters, dimensions, 0);"
        lines.append("  " + call)
    lines.append("  return true;")
    return "\n".join(lines)


def _point_selection(operation, algorithms, symbol):
    lines = []
    eligible = {
        algorithm.name:
            operation.ast.hints.requirements.get(algorithm.name, "true")
        for algorithm in algorithms
    }
    ordered = []
    for name, condition in operation.ast.hints.preferences:
        if name in eligible and name not in ordered:
            ordered.append(name)
            eligible[name] = (
                f"({eligible[name]}) && ({condition or 'true'})")
    ordered += [
        algorithm.name for algorithm in algorithms
        if algorithm.name not in ordered
    ]
    for name in ordered:
        lines.append(
            f"  if ({eligible[name]}) return {symbol}_{name}("
            "output, input_tensors, outputs, inputs, parameters, "
            "dimensions, point_index);")
    lines.append("  return true;")
    return f"""
boolean {symbol}_point(
    tensor *output, const tensor *const *input_tensors,
    void *const *outputs, const void *const *inputs,
    const void *parameters, const extent *dimensions, extent point_index) {{
{chr(10).join(_condition_bindings(operation))}
{chr(10).join(lines)}
}}
"""


def _indent(text, levels):
    prefix = "  " * levels
    return "\n".join(prefix + line for line in text.splitlines())
