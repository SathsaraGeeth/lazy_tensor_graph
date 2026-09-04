#!/usr/bin/env python3
"""Attach-only physical tensor lifecycle visualizer."""

from __future__ import annotations

import argparse
import copy
import os
import signal
import subprocess
import time
from pathlib import Path

from vtensor_lifetime import RegistryReader, Model, draw_frame, read_events, GRAPH_EVENTS


def main():
    parser = argparse.ArgumentParser(description="Passively capture physical tensor lifetimes as a GIF")
    parser.add_argument("pid", nargs="?", help="existing target PID")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="target command after --")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--poll-ms", type=int, default=25)
    parser.add_argument("--frame-ms", type=int, default=700, help="time each graph change remains visible")
    parser.add_argument("--output", default="tensor-lifetime.gif")
    parser.add_argument("--trace", default=None)
    args = parser.parse_args()

    pretraced = False
    if args.launch:
        command = ([args.pid] if args.pid else []) + args.command
        if not command:
            parser.error("use --launch -- command [args...]")
        env = os.environ.copy()
        env.pop("TENSOR_TRACE_FILE", None)
        if args.trace:
            env["TENSOR_TRACE_FILE"] = args.trace
            pretraced = True
        child = subprocess.Popen(command, env=env)
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
    model = Model()
    scenes = []
    durations = []
    position = 0
    physical = {}
    dead_physical = set()
    try:
        if not pretraced:
            os.kill(pid, signal.SIGUSR2)
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            rows, position = read_events(trace, position)
            for row in rows:
                if row["type"] in GRAPH_EVENTS:
                    model.event(row)
                    if row["type"] == "NODE_PHYSICAL":
                        owner = model.logical(int(row["parent"]))
                        if owner is not None:
                            physical[owner] = int(row["id"])
                    scenes.append((copy.deepcopy(model), dict(physical), set(dead_physical)))
                    durations.append(args.frame_ms)
            try:
                snapshot = reader.tensor_snapshot()
                materialized = False
                for raw_vtensor, tensor in snapshot.items():
                    vtensor = model.logical(raw_vtensor)
                    if vtensor is not None and tensor and physical.get(vtensor) != tensor:
                        old_tensor = physical.get(vtensor)
                        if old_tensor:
                            dead_physical.add(old_tensor)
                        physical[vtensor] = tensor
                        materialized = True
                if materialized:
                    scenes.append((copy.deepcopy(model), dict(physical), set(dead_physical)))
                    durations.append(args.frame_ms)
                removed = model.registry_update(set(snapshot))
                for ident in removed:
                    tensor = physical.get(ident)
                    if tensor:
                        dead_physical.add(tensor)
                    model.mark_dead(ident)
                    scenes.append((copy.deepcopy(model), dict(physical), set(dead_physical)))
                    durations.append(args.frame_ms)
                    model.vanish(ident)
                    physical.pop(ident, None)
                    scenes.append((copy.deepcopy(model), dict(physical), set(dead_physical)))
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
    rows, position = read_events(trace, position)
    for row in rows:
        if row["type"] in GRAPH_EVENTS:
            model.event(row)
            if row["type"] == "NODE_PHYSICAL":
                owner = model.logical(int(row["parent"]))
                if owner is not None:
                    physical[owner] = int(row["id"])
            scenes.append((copy.deepcopy(model), dict(physical), set(dead_physical)))
            durations.append(args.frame_ms)
    if not scenes:
        scenes = [(copy.deepcopy(model), dict(physical), set(dead_physical))]
        durations = [args.frame_ms]
    frames = [render(scene, scene_physical, scene_dead, index)
              for index, (scene, scene_physical, scene_dead) in enumerate(scenes)]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=durations, loop=0,
                   optimize=False, disposal=2)
    print(f"wrote {args.output} ({len(frames)} frames); physical tensors freed: {len(dead_physical)}")


def render(model, physical, dead_physical, index):
    return draw_frame(model, index, physical, dead_physical)


if __name__ == "__main__":
    main()
