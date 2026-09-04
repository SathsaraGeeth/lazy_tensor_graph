#!/usr/bin/env python3
"""Attach-only vtensor lifetime and graph visualizer.

The CSV trace supplies semantic events.  The live registry is read from the
target process through /proc/<pid>/mem, so node death is reported only when a
node has actually disappeared from the library's registry.
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import os
import signal
import struct
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


EVENT_COLORS = {
    "NODE_ALLOC": (93, 155, 232), "NODE_OP": (181, 123, 230),
    "EDGE": (62, 205, 210), "NODE_PHYSICAL": (65, 205, 170),
    "NODE_REQUEST": (245, 196, 66), "NODE_BEGIN": (255, 151, 47),
    "NODE_END": (105, 190, 105), "FREE_OBSERVED": (218, 68, 83),
}
GRAPH_EVENTS = set(EVENT_COLORS)


def registry_address(pid: int) -> int:
    exe = os.readlink(f"/proc/{pid}/exe")
    out = subprocess.check_output(["readelf", "-Ws", exe], text=True, stderr=subprocess.DEVNULL)
    value = None
    for line in out.splitlines():
        if " registry " not in f" {line} ":
            continue
        fields = line.split()
        if len(fields) >= 8 and fields[-1] == "registry":
            value = int(fields[1], 16)
            break
    if value is None:
        raise RuntimeError("could not find the tensor graph registry symbol in the target executable")

    # ELF ET_DYN executables use the load mapping's offset-zero address as the
    # relocation base.  ET_EXEC already stores the final virtual address.
    with open(f"/proc/{pid}/maps", encoding="ascii") as maps:
        for line in maps:
            fields = line.split()
            if len(fields) >= 6 and fields[5] == exe and fields[2] == "00000000":
                return int(fields[0].split("-")[0], 16) + value
    return value


class RegistryReader:
    # LP64 layout of vtensor: 10 pointers/extent fields ending at registry_next.
    # This is checked against the public C declaration in tensor_graph.h.
    NEXT_OFFSET = 80
    NODE_BYTES = 88

    def __init__(self, pid: int):
        self.pid = pid
        self.mem = open(f"/proc/{pid}/mem", "rb", buffering=0)
        self.root = registry_address(pid)

    def read(self, address: int, size: int) -> bytes:
        self.mem.seek(address)
        data = self.mem.read(size)
        if len(data) != size:
            raise OSError("short process-memory read")
        return data

    def nodes(self) -> set[int]:
        result = set()
        raw = self.read(self.root, 8)
        node = struct.unpack("<Q", raw)[0]
        while node and node not in result:
            result.add(node)
            raw = self.read(node + self.NEXT_OFFSET, 8)
            node = struct.unpack("<Q", raw)[0]
        return result

    def tensor_snapshot(self) -> dict[int, int]:
        """Return live vtensor address -> physical tensor address (zero if lazy)."""
        result = {}
        raw = self.read(self.root, 8)
        node = struct.unpack("<Q", raw)[0]
        while node and node not in result:
            result[node] = struct.unpack("<Q", self.read(node, 8))[0]
            node = struct.unpack("<Q", self.read(node + self.NEXT_OFFSET, 8))[0]
        return result

    def close(self):
        self.mem.close()


def dtype_name(value: str) -> str:
    names = {"-1": "unknown", "0": "i8", "1": "i16", "2": "i32", "3": "i64",
             "4": "u8", "5": "u16", "6": "u32", "7": "u64", "8": "f32", "9": "f64"}
    try:
        return names.get(str(int(value)), value)
    except ValueError:
        return value


def fused_name(operation: str, name: str) -> str:
    try:
        number = int(operation)
        if number & 0xF0000000 == 0xF0000000:
            return f"FUSED[{number & 0x0FFFFFFF}]"
    except ValueError:
        pass
    return name or operation


def fonts():
    candidates = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                  "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]
    for path in candidates:
        if os.path.exists(path):
            return (ImageFont.truetype(path, 16), ImageFont.truetype(path, 12),
                    ImageFont.truetype(path, 11))
    default = ImageFont.load_default()
    return default, default, default


class Model:
    def __init__(self):
        self.nodes = {}
        self.active = {}
        self.next_node = 0
        self.edges = set()
        self.deaths = set()
        self.rewires = 0
        self.last_event = "waiting for trace"
        self.process_exited = False

    def event(self, row):
        typ = row["type"]
        raw_ident = int(row["id"])
        if typ in GRAPH_EVENTS:
            for node in self.nodes.values():
                node["event"] = ""
        if typ == "NODE_ALLOC":
            previous = self.active.get(raw_ident)
            if previous is not None:
                self.deaths.add(previous)
                self.nodes[previous]["status"] = "gone"
                self.edges = {edge for edge in self.edges if previous not in edge}
            ident = self.next_node
            self.next_node += 1
            self.active[raw_ident] = ident
            self.nodes[ident] = {"name": "ALLOC", "dtype": dtype_name(row["dtype"]), "status": "live",
                                 "op": "ALLOC", "event": typ, "raw": raw_ident}
        elif typ == "NODE_OP":
            ident = self.active.get(raw_ident)
            if ident is None:
                return
            node = self.nodes[ident]
            old_edges = {edge for edge in self.edges if edge[0] == ident}
            fused = fused_name(row["operation"], row["name"])
            if old_edges:
                self.rewires += 1
                self.edges.difference_update(old_edges)
            elif fused.startswith("FUSED["):
                self.rewires += 1
            node.update(name=fused, op=row["operation"],
                        dtype=dtype_name(row["dtype"]), status="live", event=typ)
            node["edges"] = set()
        elif typ == "EDGE":
            ident = self.active.get(raw_ident)
            parent = self.active.get(int(row["parent"]))
            if ident is None or parent is None:
                return
            self.edges.add((ident, parent))
            self.nodes[ident].setdefault("edges", set()).add(parent)
            self.nodes[ident]["event"] = typ
        elif typ == "NODE_PHYSICAL":
            owner = self.active.get(int(row["parent"]))
            if owner in self.nodes:
                self.nodes[owner]["status"] = "materialized"; self.nodes[owner]["event"] = typ
        elif typ == "NODE_REQUEST":
            ident = self.active.get(raw_ident)
            if ident in self.nodes:
                self.nodes[ident]["status"] = "requested"; self.nodes[ident]["event"] = typ
        elif typ == "NODE_BEGIN":
            ident = self.active.get(raw_ident)
            if ident in self.nodes:
                self.nodes[ident]["status"] = "executing"; self.nodes[ident]["event"] = typ
        elif typ == "NODE_END":
            ident = self.active.get(raw_ident)
            if ident in self.nodes:
                self.nodes[ident]["status"] = "executed"; self.nodes[ident]["event"] = typ
        self.last_event = f"{typ} {row.get('name', '')}".strip()

    def registry_update(self, live: set[int]):
        return [ident for raw, ident in self.active.items() if raw not in live and ident not in self.deaths]

    def logical(self, raw_ident: int):
        return self.active.get(raw_ident)

    def mark_dead(self, ident: int):
        for node in self.nodes.values():
            node["event"] = ""
        self.deaths.add(ident)
        self.nodes[ident]["status"] = "dead"
        self.nodes[ident]["event"] = "FREE_OBSERVED"
        self.last_event = f"FREE observed: N{ident}"

    def vanish(self, ident: int):
        if ident in self.nodes:
            self.nodes[ident]["status"] = "gone"
            raw = self.nodes[ident].get("raw")
            if self.active.get(raw) == ident:
                self.active.pop(raw, None)
        self.edges = {edge for edge in self.edges if ident not in edge}
        self.last_event = f"DESTROYED N{ident} — node and connections removed"


def hex_color(rgb):
    return "#%02x%02x%02x" % rgb


def dot_escape(value):
    return str(value).replace('"', '\\"')


def draw_frame(model: Model, index: int, physical=None, dead_physical=None) -> Image.Image:
    visible = {ident: node for ident, node in model.nodes.items() if node.get("status") != "gone"}
    lines = [
        "digraph tensor_graph {",
        'graph [rankdir=TB, bgcolor=white, pad=0.3, nodesep=0.7, ranksep=0.9, splines=line, label="Graph", labelloc=t, fontsize=28, fontname="DejaVu Sans"];',
        'node [shape=circle, fixedsize=true, width=1.42, height=1.42, style=filled, fontname="DejaVu Sans", fontsize=11, penwidth=2.2];',
        'edge [color="#3f3f46", penwidth=1.45, arrowsize=0.62, arrowhead=vee];'
    ]
    fills = {"live": "#dcecff", "requested": "#fff2bd", "materialized": "#c9f3e8",
             "executing": "#ffd69c", "executed": "#d6f2d0", "dead": "#ffc9cf"}
    for ident, node in visible.items():
        event = node.get("event", "")
        border = hex_color(EVENT_COLORS.get(event, (55, 65, 81)))
        name = node.get("name", "?")
        if name.startswith("FUSED["):
            border = "#8b43c7"
        label = f"{name}\\nN{ident}\\n{node.get('status', 'live')}"
        lines.append(f'n{ident} [label="{dot_escape(label)}", fillcolor="{fills.get(node.get("status"), "#ffffff")}", color="{border}"];')
    for child, parent in sorted(model.edges):
        if child in visible and parent in visible:
            lines.append(f"n{parent} -> n{child};")
    if physical:
        for owner, tensor in physical.items():
            if owner not in visible or not tensor:
                continue
            dead = tensor in (dead_physical or set())
            fill = "#ffc9cf" if dead else "#e8f7ef"
            lines.append(f'p{tensor} [shape=doublecircle, width=1.15, height=1.15, label="tensor\\nT{tensor & 0xffff:x}", fillcolor="{fill}", color="#2e7d62", fontsize=10];')
            lines.append(f'n{owner} -> p{tensor} [style=dashed, color="#2e7d62", arrowsize=0.5];')
    lines.append("}")
    rendered = subprocess.run(["dot", "-Tpng"], input="\n".join(lines).encode(), stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, check=True).stdout
    graph = Image.open(io.BytesIO(rendered)).convert("RGB")
    graph.thumbnail((1380, 690), Image.Resampling.LANCZOS)
    image = Image.new("RGB", (1440, 820), "white")
    image.paste(graph, ((1440 - graph.width) // 2, 20 + (720 - graph.height) // 2))
    draw = ImageDraw.Draw(image)
    title_font, text_font, small_font = fonts()
    alive = len(visible)
    draw.text((1040, 22), f"event {index} · visible {alive} · destroyed {len(model.deaths)} · fused {model.rewires}", fill="#64748b", font=small_font)
    event_color = EVENT_COLORS.get(next((n.get("event") for n in visible.values() if n.get("event") in EVENT_COLORS), ""), (55, 65, 81))
    draw.rounded_rectangle((28, 762, 48, 782), radius=4, fill=event_color)
    draw.text((60, 762), model.last_event[:150], fill="#172033", font=text_font)
    return image


def read_events(path: Path, position: int):
    if not path.exists():
        return [], position
    fields = ["timestamp_ns", "pid", "thread", "type", "id", "parent", "operation", "dtype", "bytes", "ops", "name"]
    with path.open(newline="", encoding="utf-8") as handle:
        handle.seek(position)
        text = handle.read()
        if position == 0:
            lines = text.splitlines()
            lines = lines[1:] if lines and lines[0].startswith("timestamp_ns,") else lines
        else:
            lines = text.splitlines()
        rows = list(csv.DictReader(lines, fieldnames=fields))
        return rows, handle.tell()


def main():
    parser = argparse.ArgumentParser(description="Passively attach to a host tensor process and write a vtensor GIF")
    parser.add_argument("pid", nargs="?", help="existing target PID")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="target command after --")
    parser.add_argument("--launch", action="store_true", help="launch the command as a child")
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--poll-ms", type=int, default=25)
    parser.add_argument("--frame-ms", type=int, default=700, help="time each graph change remains visible")
    parser.add_argument("--output", default="vtensor-lifetime.gif")
    parser.add_argument("--trace", default=None, help="trace path; attach runtime default is /tmp/tensor-trace-PID.csv")
    args = parser.parse_args()
    pretraced = False
    if args.launch:
        launch_command = ([args.pid] if args.pid else []) + args.command
        if not launch_command:
            parser.error("use either PID or --launch -- command [args...]")
        child_env = os.environ.copy()
        child_env.pop("TENSOR_TRACE_FILE", None)
        if args.trace:
            child_env["TENSOR_TRACE_FILE"] = args.trace
            pretraced = True
        child = subprocess.Popen(launch_command, env=child_env)
        pid = child.pid
    elif args.pid is not None:
        try:
            pid, child = int(args.pid), None
        except ValueError:
            parser.error("PID must be an integer")
    else:
        parser.error("provide a PID or --launch -- command")
    trace = Path(args.trace or f"/tmp/tensor-trace-{pid}.csv")
    reader = RegistryReader(pid)
    model, scenes, durations, position = Model(), [], [], 0
    try:
        if not pretraced:
            os.kill(pid, signal.SIGUSR2)
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            rows, position = read_events(trace, position)
            for row in rows:
                if row["type"] in GRAPH_EVENTS:
                    model.event(row)
                    scenes.append(copy.deepcopy(model))
                    durations.append(args.frame_ms)
            try:
                removed = model.registry_update(reader.nodes())
                for ident in removed:
                    model.mark_dead(ident)
                    scenes.append(copy.deepcopy(model))
                    durations.append(args.frame_ms)
                    model.vanish(ident)
                    scenes.append(copy.deepcopy(model))
                    durations.append(args.frame_ms)
            except (OSError, ProcessLookupError):
                model.process_exited = True
                break
            time.sleep(max(args.poll_ms, 1) / 1000)
        try:
            if not pretraced:
                os.kill(pid, signal.SIGUSR2)
        except ProcessLookupError:
            model.process_exited = True
    finally:
        reader.close()
        if child is not None:
            child.wait()
    # A very short target can exit between two polls. Drain the trace once
    # more so a completed run still produces a useful graph animation.
    rows, position = read_events(trace, position)
    for row in rows:
        if row["type"] in GRAPH_EVENTS:
            model.event(row)
            scenes.append(copy.deepcopy(model))
            durations.append(args.frame_ms)
    if not scenes:
        scenes = [copy.deepcopy(model)]
        durations = [args.frame_ms]
    frames = [draw_frame(scene, index) for index, scene in enumerate(scenes)]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=durations, loop=0,
                   optimize=False, disposal=2)
    print(f"wrote {args.output} ({len(frames)} frames); exact frees observed: {len(model.deaths)}")


if __name__ == "__main__":
    main()
