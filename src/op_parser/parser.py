import pathlib
import re

from ast_nodes import (Algorithm, DeviceDef, DimensionDef, Hints, Operation,
                       ShapeDef, TypeDef, ValueDecl)
from lexer import AlgorithmBlock, Line, lex

IDENT = r"[A-Za-z_]\w*"
DTYPE = {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
         "uint64", "real32", "real64"}


class Parser:
    def __init__(self, path: pathlib.Path):
        self.path, self.items, self.at = path, lex(path), 0

    def parse(self) -> Operation:
        types, devices, dimensions, shapes = self._defines()
        name, identifier, values = self._declarations()
        algorithms = self._code()
        hints = self._hints()
        if self.at != len(self.items):
            self._error(self._line(), "unexpected content after endhints")
        return Operation(str(self.path), types, devices, dimensions, shapes,
                         name, identifier, values, algorithms, hints)

    def _defines(self):
        self._take_line("define")
        types, devices, dimensions, shapes = {}, {}, {}, {}
        while self._line().text not in ("end define", "enddefine"):
            line = self._line()
            if match := re.fullmatch(rf"type\s+({IDENT})\s*=\s*(.+);", line.text):
                name, value = match.groups()
                if value == "any":
                    members = tuple(sorted(DTYPE))
                elif value.startswith("{") and value.endswith("}"):
                    members = tuple(x.strip() for x in value[1:-1].split(","))
                else:
                    members = (value,)
                if name in types: self._error(line, f"duplicate definition '{name}'")
                types[name] = TypeDef(name, members)
            elif match := re.fullmatch(rf"devices\s+({IDENT})\s*=\s*\{{(.*)\}};", line.text):
                name, value = match.groups()
                if name in devices: self._error(line, f"duplicate definition '{name}'")
                members = tuple(x.strip() for x in value.split(",")) if value.strip() else ()
                devices[name] = DeviceDef(name, members)
            elif match := re.fullmatch(rf"dim\s+({IDENT})(?:\s*=\s*([0-9]+))?;", line.text):
                name, value = match.groups()
                if name in dimensions: self._error(line, f"duplicate definition '{name}'")
                dimensions[name] = DimensionDef(name, int(value) if value else None)
            elif match := re.fullmatch(rf"shape\s+({IDENT})\s*=\s*\[(.*)\];", line.text):
                name, value = match.groups()
                if name in shapes: self._error(line, f"duplicate definition '{name}'")
                shapes[name] = ShapeDef(name, self._shape(value, line))
            else:
                self._error(line, "invalid definition")
            self.at += 1
        self.at += 1
        return types, devices, dimensions, shapes

    def _declarations(self):
        self._take_line("declare")
        name_line = self._line()
        match = re.fullmatch(rf"name\s+({IDENT});", name_line.text)
        if not match:
            self._error(name_line, "expected 'name NAME;'")
        name = match.group(1); self.at += 1
        id_line = self._line()
        match = re.fullmatch(r"id\s+(0x[0-9a-fA-F]+|[0-9]+);", id_line.text)
        if not match:
            self._error(id_line, "expected 'id INTEGER;'")
        identifier = int(match.group(1), 0); self.at += 1
        values = []
        while self._line().text not in ("end declare", "enddeclare"):
            line = self._line()
            tensor = re.fullmatch(
                rf"(in|out)\s+({IDENT})\s*:\s*({IDENT})\s+(\[.*\]|{IDENT});",
                line.text)
            param = re.fullmatch(
                rf"param\s+({IDENT})\s*:\s*({IDENT})(?:\s*\[([0-9]+)\])?;",
                line.text)
            if tensor:
                kind, value_name, type_name, shape = tensor.groups()
                resolved = self._shape(shape[1:-1], line) if shape.startswith("[") else shape
                values.append(ValueDecl(kind, value_name, type_name, resolved))
            elif param:
                value_name, type_name, count = param.groups()
                values.append(ValueDecl("param", value_name, type_name, None,
                                        int(count) if count else 1))
            else:
                self._error(line, "invalid value declaration")
            self.at += 1
        self.at += 1
        return name, identifier, values

    def _code(self):
        self._take_line("code begin")
        algorithms = []
        while not isinstance(self._item(), Line) or self._line().text != "endcode":
            item = self._item()
            if not isinstance(item, AlgorithmBlock):
                self._error(item, "expected algorithm block")
            algorithms.append(Algorithm(item.name, item.body)); self.at += 1
        self.at += 1
        return algorithms

    def _hints(self):
        self._take_line("hints begin")
        hints = Hints()
        while self._line().text != "endhints":
            line = self._line()
            if match := re.fullmatch(r"pointwise\s*:\s*(true|false);", line.text):
                if hints.pointwise is not None: self._error(line, "duplicate pointwise hint")
                hints.pointwise = match.group(1) == "true"
            elif match := re.fullmatch(rf"target\s+({IDENT})\s*:\s*({IDENT});", line.text):
                name, target = match.groups()
                if name in hints.targets: self._error(line, f"duplicate target for '{name}'")
                hints.targets[name] = target
            elif match := re.fullmatch(rf"operations\s+({IDENT})\s*:\s*(.+);", line.text):
                name, expression = match.groups()
                if name in hints.operations:
                    self._error(line, f"duplicate operation cost for '{name}'")
                hints.operations[name] = expression
            elif match := re.fullmatch(rf"requires\s+({IDENT})\s*:\s*(.+);", line.text):
                name, expression = match.groups()
                if name in hints.requirements: self._error(line, f"duplicate requirement for '{name}'")
                hints.requirements[name] = expression
            elif match := re.fullmatch(rf"prefer\s+({IDENT})(?:\s+when\s+(.+))?;", line.text):
                hints.preferences.append(match.groups())
            else:
                self._error(line, "invalid hint")
            self.at += 1
        self.at += 1
        return hints

    def _shape(self, text, line):
        values = tuple(x.strip() for x in text.split(","))
        if not values or any(not re.fullmatch(rf"\*\*|\*|{IDENT}|[1-9][0-9]*", x) for x in values):
            self._error(line, "invalid shape")
        if "**" in values and values != ("**",):
            self._error(line, "[**] must be the complete shape")
        return values

    def _item(self):
        if self.at >= len(self.items): raise ValueError(f"{self.path}: unexpected end of file")
        return self.items[self.at]

    def _line(self):
        item = self._item()
        if not isinstance(item, Line): self._error(item, "expected DSL statement")
        return item

    def _take_line(self, text):
        line = self._line()
        if line.text != text: self._error(line, f"expected '{text}'")
        self.at += 1

    def _error(self, item, message):
        raise ValueError(f"{self.path}: line {item.number}: {message}")


def parse(path: pathlib.Path) -> Operation:
    return Parser(path).parse()
