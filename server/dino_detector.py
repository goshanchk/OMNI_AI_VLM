from __future__ import annotations

import os
from functools import lru_cache

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from common.geometry import bbox_center
from common.schemas import DetectionObject

DINO_MODEL_ID = os.getenv("DINO_MODEL_ID", "IDEA-Research/grounding-dino-base")
DINO_BOX_THRESHOLD = float(os.getenv("DINO_BOX_THRESHOLD", "0.35"))
DINO_TEXT_THRESHOLD = float(os.getenv("DINO_TEXT_THRESHOLD", "0.30"))

DEFAULT_LABELS = tuple(
    label.strip()
    for label in os.getenv(
        "DINO_LABELS",
        "person,chair,table,laptop,door,bottle,fish,cube,ball,headphones,plant,book,cup",
    ).split(",")
    if label.strip()
)
CLASS_THRESHOLDS = {
    "person": 0.42,
    "chair": 0.40,
    "table": 0.38,
    "laptop": 0.42,
    "door": 0.40,
    "window": 0.42,
    "cabinet": 0.42,
    "box": 0.38,
    "bag": 0.38,
    "bottle": 0.40,
    "keyboard": 0.46,
    "mouse": 0.46,
    "robot": 0.46,
    "fish": 0.44,
    "cube": 0.40,
    "ball": 0.40,
    "headphones": 0.44,
    "plant": 0.40,
    "book": 0.40,
    "cup": 0.40,
}


@lru_cache(maxsize=1)
def get_dino_bundle():
    processor = AutoProcessor.from_pretrained(DINO_MODEL_ID, trust_remote_code=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        DINO_MODEL_ID,
        dtype=torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()
    return processor, model


def _clip_bbox(bbox: list[int], width: int, height: int) -> list[int] | None:
    if len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _bbox_area(bbox: list[int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _bbox_iou(a: list[int], b: list[int]) -> float:
    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    return inter / max(_bbox_area(a) + _bbox_area(b) - inter, 1)


def _normalize_label(label: str) -> str:
    normalized = label.strip().lower()
    if normalized.startswith("a "):
        normalized = normalized[2:]
    if normalized.startswith("an "):
        normalized = normalized[3:]
    if normalized == "desk":
        normalized = "table"
    if normalized in {"toy fish", "goldfish"}:
        normalized = "fish"
    if normalized in {"block", "toy block", "wooden block"}:
        normalized = "cube"
    if normalized in {"sphere", "toy ball"}:
        normalized = "ball"
    if normalized in {"headset", "earmuffs"}:
        normalized = "headphones"
    if normalized in {"potted plant", "flower pot", "flower"}:
        normalized = "plant"
    if normalized in {"textbook", "notebook book"}:
        normalized = "book"
    if normalized in {"paper cup", "cone cup"}:
        normalized = "cup"
    return normalized


def _is_reasonable(label: str, bbox: list[int], width: int, height: int) -> bool:
    area_ratio = _bbox_area(bbox) / float(max(width * height, 1))
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    aspect = bbox_w / max(bbox_h, 1)
    touches_border = bbox[0] <= 1 or bbox[1] <= 1 or bbox[2] >= width - 2 or bbox[3] >= height - 2

    if area_ratio < 0.002:
        return False
    if label not in {"door", "window"} and area_ratio > 0.55:
        return False
    if label in {"keyboard", "mouse", "bottle"} and area_ratio > 0.12:
        return False
    if label == "fish" and (area_ratio < 0.003 or area_ratio > 0.18):
        return False
    if label == "fish" and (aspect < 0.35 or aspect > 2.8):
        return False
    if label == "cube" and (area_ratio < 0.003 or area_ratio > 0.20):
        return False
    if label == "cube" and (aspect < 0.6 or aspect > 1.7):
        return False
    if label == "ball" and (area_ratio < 0.002 or area_ratio > 0.16):
        return False
    if label == "ball" and (aspect < 0.6 or aspect > 1.6):
        return False
    if label == "headphones" and (area_ratio < 0.01 or area_ratio > 0.35):
        return False
    if label == "plant" and (area_ratio < 0.003 or area_ratio > 0.25):
        return False
    if label == "book" and (area_ratio < 0.01 or area_ratio > 0.35):
        return False
    if label == "cup" and (area_ratio < 0.002 or area_ratio > 0.16):
        return False
    if label in {"keyboard", "mouse", "bottle", "bag", "box", "robot", "cube", "cup"} and touches_border:
        return False
    if label == "person" and area_ratio > 0.65:
        return False
    if label == "person" and bbox_h < height * 0.20:
        return False
    if label == "laptop" and (aspect < 0.8 or aspect > 3.0):
        return False
    if label == "door" and aspect > 1.2:
        return False
    if label == "chair" and (aspect < 0.25 or aspect > 1.8):
        return False
    if label == "table" and bbox_w < width * 0.12:
        return False
    if label == "window" and area_ratio < 0.03:
        return False
    return True


def _dedupe(detections: list[DetectionObject]) -> list[DetectionObject]:
    kept: list[DetectionObject] = []
    for det in sorted(detections, key=lambda item: item.confidence or 0.0, reverse=True):
        duplicate = False
        for existing in kept:
            if det.label == existing.label and _bbox_iou(det.bbox_2d, existing.bbox_2d) > 0.6:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def run_dino(
    image: Image.Image,
    preferred_label: str | None = None,
    requested_labels: list[str] | None = None,
) -> list[DetectionObject]:
    processor, model = get_dino_bundle()
    labels = [preferred_label] if preferred_label else list(requested_labels or DEFAULT_LABELS)
    width, height = image.size
    detections: list[DetectionObject] = []
    model_param = next(model.parameters())
    model_device = model_param.device
    model_dtype = model_param.dtype

    for label in labels:
        text_labels = [[f"a {label}"]]
        inputs = processor(images=image, text=text_labels, return_tensors="pt")
        prepared_inputs = {}
        for key, value in inputs.items():
            if not hasattr(value, "to"):
                prepared_inputs[key] = value
                continue
            moved = value.to(model_device)
            if torch.is_floating_point(moved):
                moved = moved.to(model_dtype)
            prepared_inputs[key] = moved
        inputs = prepared_inputs

        with torch.inference_mode():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=CLASS_THRESHOLDS.get(label, DINO_BOX_THRESHOLD),
            text_threshold=DINO_TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]],
        )
        result = results[0]

        detected_labels = result.get("text_labels") or result.get("labels") or []
        for box, score, detected_label in zip(result["boxes"], result["scores"], detected_labels):
            normalized_label = _normalize_label(str(detected_label))
            confidence = float(score.item())
            if confidence < CLASS_THRESHOLDS.get(normalized_label, DINO_BOX_THRESHOLD):
                continue

            clipped_bbox = _clip_bbox([int(round(v)) for v in box.tolist()], width, height)
            if clipped_bbox is None or not _is_reasonable(normalized_label, clipped_bbox, width, height):
                continue

            detections.append(
                DetectionObject(
                    label=normalized_label,
                    bbox_2d=clipped_bbox,
                    center_2d=bbox_center(clipped_bbox),
                    confidence=confidence,
                    source="dino",
                )
            )

    deduped = _dedupe(detections)
    if preferred_label:
        return deduped[:1]
    return deduped
