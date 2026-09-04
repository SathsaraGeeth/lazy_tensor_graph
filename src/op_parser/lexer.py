from dataclasses import dataclass
import pathlib
import re


@dataclass(frozen=True)
class Line:
    number: int
    text: str


@dataclass(frozen=True)
class AlgorithmBlock:
    number: int
    name: str
    body: str


Item = Line | AlgorithmBlock
OPEN_RE = re.compile(r"algorithm\s+([A-Za-z_]\w*)\s+begin$")
END_RE = re.compile(r"endalgorithm\s*:\s*([A-Za-z_]\w*)$")


def lex(path: pathlib.Path) -> list[Item]:
    raw = path.read_text(encoding="utf-8").splitlines()
    items: list[Item] = []
    index = 0
    while index < len(raw):
        number, stripped = index + 1, raw[index].strip()
        index += 1
        if not stripped or stripped.startswith("//"):
            continue
        opening = OPEN_RE.fullmatch(stripped)
        if not opening:
            items.append(Line(number, stripped))
            continue
        name, body = opening.group(1), []
        while index < len(raw):
            closing = END_RE.fullmatch(raw[index].strip())
            if closing:
                if closing.group(1) != name:
                    raise ValueError(
                        f"line {index + 1}: expected 'endalgorithm: {name}'")
                index += 1
                break
            body.append(raw[index])
            index += 1
        else:
            raise ValueError(f"line {number}: unterminated algorithm '{name}'")
        if not "\n".join(body).strip():
            raise ValueError(f"line {number}: empty algorithm '{name}'")
        items.append(AlgorithmBlock(number, name, "\n".join(body)))
    return items

