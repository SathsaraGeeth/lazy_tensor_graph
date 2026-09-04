import pathlib
import re

CTYPE = {"int8":"int8","int16":"int16","int32":"int32","int64":"int64",
         "uint8":"uint8","uint16":"uint16","uint32":"uint32","uint64":"uint64",
         "real32":"real32","real64":"real64"}
SIZE = {"int8":1,"uint8":1,"int16":2,"uint16":2,"int32":4,"uint32":4,
        "real32":4,"int64":8,"uint64":8,"real64":8}


def _cost_case(operation) -> str:
    expressions = list(operation.ast.hints.operations.values())
    expressions += list(operation.ast.hints.requirements.values())
    expressions += [
        condition for _, condition in operation.ast.hints.preferences
        if condition is not None
    ]
    referenced = set(re.findall(
        r"\b[A-Za-z_]\w*\b", " ".join(expressions)))
    lines = [
        f"    case 0x{operation.ast.id:08x}u: {{",
        f"      if (!output || input_count != {len(operation.inputs)}) return 0;",
        "      (void)parameters;",
    ]
    for symbol, definition in operation.ast.dimensions.items():
        if symbol not in referenced:
            continue
        if definition.value is not None:
            expression = str(definition.value)
        else:
            expression = None
            for index, value in enumerate(operation.inputs):
                if symbol in value.shape:
                    expression = (
                        f"inputs[{index}]->shape[{value.shape.index(symbol)}]")
                    break
            if expression is None:
                for value in operation.outputs:
                    if symbol in value.shape:
                        expression = (
                            f"output->shape[{value.shape.index(symbol)}]")
                        break
        if expression is not None:
            lines.append(f"      extent {symbol} = {expression};")
    if "point_count" in referenced:
        lines.append("      extent point_count = output->size;")
    for index, value in enumerate(operation.inputs):
        if f"{value.name}_size" in referenced:
            lines.append(
                f"      extent {value.name}_size = inputs[{index}]->size;")
        if f"{value.name}_rank" in referenced:
            lines.append(
                f"      extent {value.name}_rank = inputs[{index}]->rank;")
    for value in operation.outputs:
        if f"{value.name}_size" in referenced:
            lines.append(f"      extent {value.name}_size = output->size;")
        if f"{value.name}_rank" in referenced:
            lines.append(f"      extent {value.name}_rank = output->rank;")
    offset = 0
    for value in operation.parameters:
        alignment = SIZE[value.type_key]
        offset = (offset + alignment - 1) // alignment * alignment
        ctype = CTYPE[value.type_key]
        if value.name not in referenced:
            offset += SIZE[value.type_key] * value.count
            continue
        if value.count > 1:
            lines.append(
                f"      const {ctype} *{value.name} = "
                f"(const {ctype} *)((const uint8 *)parameters + {offset});")
        else:
            lines.append(
                f"      {ctype} {value.name} = *(const {ctype} *)"
                f"((const uint8 *)parameters + {offset});")
        offset += SIZE[value.type_key] * value.count
    eligible = {
        algorithm.name:
            operation.ast.hints.requirements.get(algorithm.name, "true")
        for algorithm in operation.ast.algorithms
    }
    ordered = []
    for name, condition in operation.ast.hints.preferences:
        if name in eligible and name not in ordered:
            ordered.append(name)
            eligible[name] = (
                f"({eligible[name]}) && ({condition or 'true'})")
    ordered += [
        algorithm.name for algorithm in operation.ast.algorithms
        if algorithm.name not in ordered
    ]
    for name in ordered:
        lines.append(
            f"      if ({eligible[name]}) return (uint64)"
            f"({operation.ast.hints.operations[name]});")
    lines += ["      return 0;", "    }"]
    return "\n".join(lines)


def emit_registry(bitcode: pathlib.Path, records, output: pathlib.Path) -> None:
    data = bitcode.read_bytes()
    array = ", ".join(f"0x{byte:02x}" for byte in data)
    grouped = {}
    for op, out_dtype, in_dtype, symbol in records:
        grouped.setdefault(op.ast.id, []).append((out_dtype, in_dtype, symbol))
    cases = "\n".join(
        f"    case 0x{identifier:08x}u:\n" +
        "\n".join(
            f"      if (output_dtype == {out_dtype} && input_dtype == {in_dtype}) "
            f"{{ *name = \"{symbol}\"; return false; }}"
            for out_dtype, in_dtype, symbol in variants) +
        "\n      return true;"
        for identifier, variants in grouped.items())
    unique_operations = list(
        dict((record[0].ast.id, record[0]) for record in records).values())
    pointwise = "\n".join(
        f"    case 0x{op.ast.id:08x}u: return {'true' if op.ast.hints.pointwise else 'false'};"
        for op in unique_operations)
    costs = "\n".join(_cost_case(op) for op in unique_operations)
    source = f"""/* Generated operation module registry; do not edit. */
#include "tensor_core.h"

static const uint8 tensor_operations_bc[] = {{
{array}
}};

boolean tensor_jit_bitcode_find(
    uint32 id, dtype_t output_dtype, dtype_t input_dtype,
    const char **name, const uint8 **data, extent *size) {{
  if (!name || !data || !size) return true;
  *data = tensor_operations_bc;
  *size = sizeof(tensor_operations_bc);
  switch (id) {{
{cases}
    default: return true;
  }}
}}

boolean tensor_jit_is_pointwise(uint32 id) {{
  switch (id) {{
{pointwise}
    default: return false;
  }}
}}

boolean tensor_profile_has_operation_count(uint32 id) {{
  switch (id) {{
{chr(10).join(f"    case 0x{op.ast.id:08x}u: return true;" for op in unique_operations)}
    default: return false;
  }}
}}

uint64 tensor_profile_operation_count(
    uint32 id, const tensor *output, const tensor *const *inputs,
    extent input_count, const void *parameters) {{
  switch (id) {{
{costs}
    default: return 0;
  }}
}}
"""
    output.write_text(source, encoding="utf-8")
