from __future__ import annotations

import argparse
import logging
import sys

from server.task_parser import parse_task_prompt
from server.zmq.state import SharedState

logger = logging.getLogger("hoverai.zmq_receiver")

LABEL_KEYS: dict[int, str | None] = {
    ord("0"): None,
    ord("p"): "person",
    ord("c"): "cube",
    ord("b"): "ball",
    ord("h"): "headphones",
    ord("f"): "fish",
    ord("r"): "robot",
    ord("t"): "table",
    ord("l"): "laptop",
    ord("d"): "door",
    ord("n"): "plant",
    ord("k"): "book",
    ord("u"): "cup",
    ord("e"): "bottle",
}

PROJECTION_ALIASES: dict[str, str] = {
    "wall": "wall",
    "screen": "drone_screen",
    "drone_screen": "drone_screen",
    "drone-screen": "drone_screen",
}


def normalize_label(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower().replace("-", " ").replace("_", " ")
    return " ".join(normalized.split()) or None


def normalize_projection(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower().replace(" ", "_")
    return PROJECTION_ALIASES.get(normalized)


def compose_runtime_prompt(runtime_label: str | None, runtime_projection: str | None) -> str | None:
    if runtime_label and runtime_projection == "wall":
        return f"find {runtime_label} and project on wall"
    if runtime_label and runtime_projection == "drone_screen":
        return f"find {runtime_label} and project on drone screen"
    if runtime_label:
        return f"find {runtime_label}"
    if runtime_projection == "wall":
        return "project on wall"
    if runtime_projection == "drone_screen":
        return "project on drone screen"
    return None


def current_runtime_prompt(state: SharedState, args: argparse.Namespace) -> str | None:
    return state.runtime_prompt if state.runtime_prompt is not None else args.task_prompt


def current_runtime_label(state: SharedState, args: argparse.Namespace) -> str | None:
    return state.runtime_label if state.runtime_label is not None else args.preferred_label


def current_prefer_wall(state: SharedState, args: argparse.Namespace) -> bool:
    if state.runtime_projection == "wall":
        return True
    if state.runtime_projection == "drone_screen":
        return False
    return bool(args.prefer_wall)


def effective_projection(state: SharedState, args: argparse.Namespace) -> str | None:
    parsed_task = parse_task_prompt(current_runtime_prompt(state, args))
    if state.runtime_projection is not None:
        return state.runtime_projection
    if parsed_task.projection_target is not None:
        return parsed_task.projection_target
    return "wall" if args.prefer_wall else None


def runtime_status_line(state: SharedState, args: argparse.Namespace) -> str:
    prompt = current_runtime_prompt(state, args) or "-"
    label = current_runtime_label(state, args) or "all"
    projection = effective_projection(state, args) or "-"
    return f"label={label} | projection={projection} | prompt={prompt}"


def set_runtime_command_hint(state: SharedState, message: str) -> None:
    state.command_hint = message[:180]
    logger.info("runtime_command: %s", message)


def print_runtime_help() -> None:
    help_text = (
        "\nRuntime commands:\n"
        "  help                         Show this help\n"
        "  status                       Show active runtime prompt\n"
        "  object <label>               Set target object, e.g. object person\n"
        "  projection <wall|screen>     Set projection target\n"
        "  prompt <text>                Set full prompt manually\n"
        "  clear object                 Clear runtime object filter\n"
        "  clear projection             Clear runtime projection override\n"
        "  clear prompt                 Clear runtime prompt override\n"
        "  reset                        Clear all runtime overrides\n"
        "\nExamples:\n"
        "  object person\n"
        "  projection wall\n"
        "  projection screen\n"
        "  prompt find cube and project on wall\n"
    )
    print(help_text, flush=True)


def apply_keyboard_label(state: SharedState, label: str | None) -> None:
    state.runtime_label = label
    state.runtime_prompt = compose_runtime_prompt(state.runtime_label, state.runtime_projection)
    state.command_hint = f"keyboard target set to {state.runtime_label or 'all'}"


def stdin_command_loop(state: SharedState, args: argparse.Namespace) -> None:
    if not sys.stdin or not sys.stdin.isatty():
        return

    print_runtime_help()
    while not state.stop:
        try:
            line = input("hoverai> ").strip()
        except EOFError:
            return
        except KeyboardInterrupt:
            state.stop = True
            return

        if not line:
            continue

        lower = line.lower()
        with state.lock:
            if lower == "help":
                set_runtime_command_hint(state, "help shown in terminal")
                print("", flush=True)
                print_runtime_help()
                continue

            if lower == "status":
                status_line = runtime_status_line(state, args)
                set_runtime_command_hint(state, status_line)
                print(status_line, flush=True)
                continue

            if lower == "reset":
                state.runtime_label = None
                state.runtime_projection = None
                state.runtime_prompt = None
                set_runtime_command_hint(state, "runtime overrides cleared")
                print("runtime overrides cleared", flush=True)
                continue

            if lower == "clear object":
                state.runtime_label = None
                state.runtime_prompt = compose_runtime_prompt(state.runtime_label, state.runtime_projection)
                set_runtime_command_hint(state, "runtime object cleared")
                print("runtime object cleared", flush=True)
                continue

            if lower == "clear projection":
                state.runtime_projection = None
                state.runtime_prompt = compose_runtime_prompt(state.runtime_label, state.runtime_projection)
                set_runtime_command_hint(state, "runtime projection cleared")
                print("runtime projection cleared", flush=True)
                continue

            if lower == "clear prompt":
                state.runtime_prompt = compose_runtime_prompt(state.runtime_label, state.runtime_projection)
                set_runtime_command_hint(state, "runtime prompt rebuilt from object/projection")
                print(f"runtime prompt: {state.runtime_prompt or '-'}", flush=True)
                continue

            if lower.startswith("object "):
                label = normalize_label(line[7:])
                if not label:
                    message = "object label cannot be empty"
                    set_runtime_command_hint(state, message)
                    print(message, flush=True)
                    continue
                state.runtime_label = label
                state.runtime_prompt = compose_runtime_prompt(state.runtime_label, state.runtime_projection)
                message = f"runtime object set to {label}"
                set_runtime_command_hint(state, message)
                print(f"{message}; prompt={state.runtime_prompt or '-'}", flush=True)
                continue

            if lower.startswith("projection "):
                projection = normalize_projection(line[len("projection "):])
                if projection is None:
                    message = "projection must be wall or screen"
                    set_runtime_command_hint(state, message)
                    print(message, flush=True)
                    continue
                state.runtime_projection = projection
                state.runtime_prompt = compose_runtime_prompt(state.runtime_label, state.runtime_projection)
                message = f"runtime projection set to {projection}"
                set_runtime_command_hint(state, message)
                print(f"{message}; prompt={state.runtime_prompt or '-'}", flush=True)
                continue

            if lower.startswith("prompt "):
                prompt = line[len("prompt "):].strip()
                if not prompt:
                    message = "prompt text cannot be empty"
                    set_runtime_command_hint(state, message)
                    print(message, flush=True)
                    continue
                state.runtime_prompt = prompt
                parsed = parse_task_prompt(prompt)
                state.runtime_projection = parsed.projection_target
                state.runtime_label = parsed.target_label
                message = f"runtime prompt set to {prompt}"
                set_runtime_command_hint(state, message)
                print(runtime_status_line(state, args), flush=True)
                continue

            message = f"unknown command: {line}"
            set_runtime_command_hint(state, message)
            print(message, flush=True)
