from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParsedTask:
    prompt: str | None
    labels: list[str]
    target_label: str | None
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
    "fish": ("fish", "toy fish", "goldfish"),
    "cube": ("cube", "block", "toy block", "wooden block"),
    "ball": ("ball", "sphere", "toy ball"),
    "headphones": ("headphones", "headset", "earmuffs"),
    "plant": ("plant", "potted plant", "flower pot", "flower"),
    "book": ("book", "notebook book", "textbook"),
    "cup": ("cup", "paper cup", "cone cup"),
}

TARGET_ACTION_MARKERS = (
    "find",
    "locate",
    "approach",
    "fly to",
    "go to",
    "move to",
    "track",
    "follow",
    "target",
)

PROJECTION_WALL_MARKERS = (
    "wall",
    "project on wall",
    "projection wall",
    "on wall",
)

PROJECTION_DRONE_SCREEN_MARKERS = (
    "drone screen",
    "screen on drone",
    "onboard screen",
    "screen mounted on drone",
    "project on drone screen",
)


def _find_matches(lowered: str) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    seen: set[str] = set()
    for canonical, synonyms in LABEL_SYNONYMS.items():
        positions = [lowered.find(token) for token in synonyms if lowered.find(token) != -1]
        if not positions or canonical in seen:
            continue
        matches.append((min(positions), canonical))
        seen.add(canonical)
    matches.sort(key=lambda item: item[0])
    return matches


def _pick_target_label(lowered: str, matched_labels: list[str], projection_target: str | None) -> str | None:
    for marker in TARGET_ACTION_MARKERS:
        marker_pos = lowered.find(marker)
        if marker_pos == -1:
            continue
        suffix = lowered[marker_pos + len(marker) :]
        for canonical, synonyms in LABEL_SYNONYMS.items():
            if projection_target == "wall" and canonical == "wall":
                continue
            if any(token in suffix[:80] for token in synonyms):
                return canonical

    for label in matched_labels:
        if projection_target == "wall" and label == "wall":
            continue
        return label
    return "wall" if projection_target == "wall" and "wall" in matched_labels else None


def parse_task_prompt(task_prompt: str | None) -> ParsedTask:
    if not task_prompt:
        return ParsedTask(prompt=None, labels=[], target_label=None, projection_target=None)

    lowered = task_prompt.lower()
    matched_labels = [label for _, label in _find_matches(lowered)]

    projection_target = None
    if any(token in lowered for token in PROJECTION_DRONE_SCREEN_MARKERS):
        projection_target = "drone_screen"
    elif any(token in lowered for token in PROJECTION_WALL_MARKERS):
        projection_target = "wall"

    target_label = _pick_target_label(lowered, matched_labels, projection_target)

    labels: list[str] = []
    if target_label:
        labels.append(target_label)
    for label in matched_labels:
        if label not in labels:
            labels.append(label)
    if projection_target == "wall" and "wall" not in labels:
        labels.append("wall")

    return ParsedTask(
        prompt=task_prompt,
        labels=labels,
        target_label=target_label,
        projection_target=projection_target,
    )
