#!/usr/bin/env python3
"""Compile .op sources into one LLVM bitcode module"""

import argparse
import pathlib
import sys

from ir import emit_bitcode
from parser import parse
from registry import emit_registry
from validator import validate, validate_tree


def main() -> int:
    arguments = argparse.ArgumentParser()
    arguments.add_argument("input", type=pathlib.Path)
    arguments.add_argument("output", type=pathlib.Path)
    arguments.add_argument("--device", choices=("generic", "cpu", "cuda"),
                           default="generic")
    args = arguments.parse_args()
    try:
        operations = [validate(parse(path)) for path in sorted(args.input.rglob("*.op"))]
        validate_tree(operations)
        args.output.mkdir(parents=True, exist_ok=True)
        bitcode = args.output / "operations.bc"
        records = emit_bitcode(operations, bitcode, args.device)
        emit_registry(bitcode, records, args.output / "registry.c")
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
