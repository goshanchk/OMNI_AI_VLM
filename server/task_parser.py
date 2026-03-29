from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParsedTask:
    prompt: str | None
    labels: list[str]
    projection_target: str | None


LABEL_SYNONYMS = {
    "person": ("person", "human", "man", "woman", "people"),
    "chair": ("chair", "seat", "stool"),
    "table": ("table", "desk"),
    "laptop": ("laptop", "notebook"),
    "door": ("door", "doorway", "entrance"),
    "wall": ("wall",),
    "robot": ("robot", "rover", "platform"),
    "keyboard": ("keyboard",),
    "mouse": ("mouse",),
    "window": ("window",),
    "cabinet": ("cabinet", "shelf"),
    "box": ("box",),
    "bag": ("bag", "backpack"),
    "bottle": ("bottle",),
}


def parse_task_prompt(task_prompt: str | None) -> ParsedTask:
    if not task_prompt:
        return ParsedTask(prompt=None, labels=[], projection_target=None)

    lowered = task_prompt.lower()
    labels: list[str] = []
    for canonical, synonyms in LABEL_SYNONYMS.items():
        if any(token in lowered for token in synonyms):
            labels.append(canonical)

    projection_target = None
    if any(token in lowered for token in ("wall", "project on wall", "projection wall")):
        projection_target = "wall"
    elif any(
        token in lowered
        for token in ("drone screen", "screen on drone", "onboard screen", "screen mounted on drone", "project on drone screen")
    ):
        projection_target = "drone_screen"

    return ParsedTask(prompt=task_prompt, labels=labels, projection_target=projection_target)
