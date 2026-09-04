"""AST nodes for the tensor operation language"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TypeDef:
    name: str
    dtypes: tuple[str, ...]


@dataclass(frozen=True)
class DeviceDef:
    name: str
    devices: tuple[str, ...]


@dataclass(frozen=True)
class DimensionDef:
    name: str
    value: int | None


@dataclass(frozen=True)
class ShapeDef:
    name: str
    dimensions: tuple[str, ...]


@dataclass(frozen=True)
class ValueDecl:
    kind: str
    name: str
    type_name: str
    shape: tuple[str, ...] | str | None
    count: int = 1


@dataclass(frozen=True)
class Algorithm:
    name: str
    body: str


@dataclass
class Hints:
    pointwise: bool | None = None
    targets: dict[str, str] = field(default_factory=dict)
    operations: dict[str, str] = field(default_factory=dict)
    requirements: dict[str, str] = field(default_factory=dict)
    preferences: list[tuple[str, str | None]] = field(default_factory=list)


@dataclass
class Operation:
    source: str
    types: dict[str, TypeDef]
    devices: dict[str, DeviceDef]
    dimensions: dict[str, DimensionDef]
    shapes: dict[str, ShapeDef]
    name: str
    id: int
    values: list[ValueDecl]
    algorithms: list[Algorithm]
    hints: Hints
