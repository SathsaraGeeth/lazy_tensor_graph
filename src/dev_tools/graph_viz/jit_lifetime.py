#!/usr/bin/env python3
"""Show runtime JIT cache activity and persistent-cache reuse."""

from __future__ import annotations

import argparse
import copy
import csv
import io
import os
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


EVENTS = {
    "JIT_LOOKUP_BEGIN": (92, 119, 194),
    "JIT_CACHE_MISS": (231, 150, 54),
    "JIT_CACHE_HIT": (67, 171, 99),
    "JIT_MATERIALIZE_BEGIN": (226, 176, 55),
    "JIT_MATERIALIZE_END": (68, 153, 196),
    "PERSISTENT_LOAD": (126, 91, 179),
    "PERSISTENT_STORE": (126, 91, 179),
}
FIELDS = ["timestamp_ns", "pid", "thread", "type", "id", "parent", "operation", "dtype", "bytes", "ops", "name"]


def font(size):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(path, size) if os.path.exists(path) else ImageFont.load_default()


def dtype_name(value):
    return {"8": "f32", "9": "f64", "4": "u8", "5": "u16"}.get(str(value), str(value))


def operation_name(value):
    number = int(value)
    if number & 0xF0000000 == 0xF0000000:
        return f"FUSED[{number & 0x0FFFFFFF}]"
    names = {0x10001: "BRIGHTNESS", 0x30003: "CAST", 0x40001: "MATRIX_MUL", 0x40003: "CONV2D"}
    return names.get(number, f"OP 0x{number:x}")


def key_label(row):
    return f"{operation_name(row['operation'])}\\n{dtype_name(row['dtype'])}"


def read_trace(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row["type"] in EVENTS and row["type"].startswith("JIT_")]


class Model:
    def __init__(self):
        self.keys = {}
        self.runs = []
        self.edges = set()
        self.disk = False
        self.last = "waiting for JIT activity"
        self.event = ""

    def jit_event(self, row, run):
        key = (row["operation"], row["dtype"])
        item = self.keys.setdefault(key, {"label": key_label(row), "state": "lookup", "run": run})
        item["run"] = run
        item["state"] = {
            "JIT_LOOKUP_BEGIN": "lookup", "JIT_CACHE_MISS": "miss", "JIT_CACHE_HIT": "hit",
            "JIT_MATERIALIZE_BEGIN": "compiling", "JIT_MATERIALIZE_END": "ready",
        }[row["type"]]
        item["event"] = row["type"]
        self.event = row["type"]
        self.last = f"run {run}: {row['type']} · {item['label'].replace(chr(92) + 'n', ' / ')}"
        key_id = f"k{row['operation']}_{row['dtype']}"
        self.edges.add((f"run{run}", key_id))
        self.edges.add((key_id, "ram"))

    def persistent(self, kind, run):
        self.disk = True
        self.event = kind
        self.last = f"run {run}: persistent cache {kind.removeprefix('PERSISTENT_').lower()} observed"
        self.edges.add(("ram", "disk") if kind == "PERSISTENT_STORE" else ("disk", f"run{run}"))


def render(model, index):
    lines = [
        "digraph jit_graph {",
        'graph [rankdir=TB, bgcolor=white, pad=0.3, nodesep=0.75, ranksep=0.9, splines=line, label="JIT CACHE", labelloc=t, fontsize=28, fontname="DejaVu Sans"];',
        'node [shape=circle, fixedsize=true, width=1.45, height=1.45, style=filled, fontname="DejaVu Sans", fontsize=11, penwidth=2.2];',
        'edge [color="#3f3f46", penwidth=1.45, arrowsize=0.6, arrowhead=vee];',
        'ram [label="RAM\\ncache", fillcolor="#dcecff", color="#405b86"];',
        f'disk [shape=doublecircle, label="persistent\\ncache", fillcolor="{("#e7dafa" if model.disk else "#f5f5f5")}", color="#7e57a2"];',
    ]
    for run in model.runs:
        lines.append(f'run{run} [label="run {run}", fillcolor="#eeeeee", color="#52525b"];')
    fills = {"lookup": "#dcecff", "miss": "#ffd79b", "compiling": "#ffe99d", "hit": "#d5f2d4", "ready": "#c9f3e8"}
    borders = {"JIT_LOOKUP_BEGIN": "#5c77c2", "JIT_CACHE_MISS": "#e79636", "JIT_CACHE_HIT": "#43ab63",
               "JIT_MATERIALIZE_BEGIN": "#e2b037", "JIT_MATERIALIZE_END": "#4499c4"}
    for (operation, dtype), item in model.keys.items():
        ident = f"k{operation}_{dtype}"
        border = borders.get(item.get("event"), "#52525b")
        label = item["label"] + "\\n" + item["state"]
        lines.append(f'{ident} [label="{label}", fillcolor="{fills[item["state"]]}", color="{border}"];')
    for left, right in sorted(model.edges):
        lines.append(f"{left} -> {right};")
    lines.append("}")
    png = subprocess.run(["dot", "-Tpng"], input="\n".join(lines).encode(), stdout=subprocess.PIPE, check=True).stdout
    graph = Image.open(io.BytesIO(png)).convert("RGB")
    graph.thumbnail((1380, 720), Image.Resampling.LANCZOS)
    image = Image.new("RGB", (1440, 820), "white")
    image.paste(graph, ((1440 - graph.width) // 2, 18 + (720 - graph.height) // 2))
    draw = ImageDraw.Draw(image)
    draw.text((1040, 22), f"event {index} · keys {len(model.keys)}", fill="#64748b", font=font(11))
    color = EVENTS.get(model.event, (55, 65, 81))
    draw.rounded_rectangle((28, 762, 48, 782), radius=4, fill=color)
    draw.text((60, 762), model.last[:150], fill="#172033", font=font(12))
    return image


def main():
    parser = argparse.ArgumentParser(description="Capture JIT RAM-cache and persistent-cache reuse as a GIF")
    parser.add_argument("--launch", action="store_true", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    parser.add_argument("--output", default="jit_lifecycle.gif")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--frame-ms", type=int, default=900)
    args = parser.parse_args()
    command = [part for part in args.command if part != "--"]
    if not command:
        parser.error("use --launch -- command [args...]")
    cache = args.cache or f"/tmp/tensor-jit-graph-viz-{os.getpid()}.cache"
    model, scenes, durations = Model(), [], []
    for run in (1, 2):
        trace = Path(f"/tmp/tensor-jit-graph-viz-{os.getpid()}-{run}.csv")
        before = Path(cache).stat() if Path(cache).exists() else None
        child_env = os.environ.copy()
        child_env["TENSOR_TRACE_FILE"] = str(trace)
        child_env["TENSOR_JIT_CACHE_PATH"] = cache
        child = subprocess.Popen(command, env=child_env)
        model.runs.append(run)
        if before:
            model.persistent("PERSISTENT_LOAD", run)
            scenes.append(copy.deepcopy(model)); durations.append(args.frame_ms)
        while child.poll() is None:
            time.sleep(0.01)
        child.wait()
        for row in read_trace(trace):
            model.jit_event(row, run)
            scenes.append(copy.deepcopy(model)); durations.append(args.frame_ms)
        after = Path(cache).stat() if Path(cache).exists() else None
        if after and (before is None or after.st_mtime_ns != before.st_mtime_ns or after.st_size != before.st_size):
            model.persistent("PERSISTENT_STORE", run)
            scenes.append(copy.deepcopy(model)); durations.append(args.frame_ms)
    if not scenes:
        scenes = [copy.deepcopy(model)]; durations = [args.frame_ms]
    frames = [render(scene, i) for i, scene in enumerate(scenes)]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=durations, loop=0,
                   optimize=False, disposal=2)
    print(f"wrote {args.output} ({len(frames)} frames); cache={cache}")


if __name__ == "__main__":
    main()
