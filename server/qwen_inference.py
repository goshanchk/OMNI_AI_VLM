from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

import torch
from PIL import Image, ImageFilter, ImageOps
from transformers import AutoProcessor

from common.geometry import bbox_center
from common.schemas import DetectionObject, ProjectionSurface, ProjectionWall, VisionResult

try:
    from transformers import AutoModelForImageTextToText as VisionModelClass
except ImportError:
    from transformers import AutoModelForVision2Seq as VisionModelClass


MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "512"))
ALLOWED_LABELS = ("person", "chair", "table", "laptop", "door", "wall")


def _build_prompt(
    image_width: int,
    image_height: int,
    preferred_label: str | None,
    task_prompt: str | None,
    requested_labels: list[str] | None,
    projection_target: str | None,
) -> str:
    preferred_instruction = ""
    if preferred_label:
        preferred_instruction = (
            f"\nPriority target label: {preferred_label}. "
            f"If that label is visible, make its bbox especially tight and accurate."
        )
    requested_instruction = f"\nRequested task labels: {', '.join(requested_labels)}." if requested_labels else ""
    projection_instruction = f"\nPreferred projection target: {projection_target}." if projection_target else ""
    task_instruction = f"\nUser task prompt: {task_prompt}" if task_prompt else ""

    return f"""
You are a precise vision grounding module for an indoor drone.

Return ONLY valid JSON and nothing else.

Image size:
- width: {image_width}
- height: {image_height}

Allowed labels only:
{", ".join(ALLOWED_LABELS)}
{preferred_instruction}
{requested_instruction}
{projection_instruction}
{task_instruction}

Rules:
1. Detect only clearly visible objects. If uncertain, omit the object.
1a. Focus on the objects relevant to the task prompt.
2. Bounding boxes must be tight around the visible object extent.
3. All coordinates must be integer pixel coordinates in this image.
4. Use [x1, y1, x2, y2] with 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height.
5. Do not invent objects.
6. Do not use a huge box covering most of the image unless the object truly occupies most of it.
7. Choose the best projection surface for this task: wall or drone_screen.
8. A valid projection surface must be visible, reasonably front-facing, and as free/empty as possible.
9. Prefer a blank wall patch if projecting to wall.
10. If projecting to the drone's own onboard screen, set surface_type to "drone_screen" and do not invent any external screen or monitor in the scene.
11. The bbox for projection surface must cover only the usable projection region. For drone_screen, bbox_2d may be empty.
12. If no confident instance exists, return an empty objects list.

Output schema:
{{
  "scene_description": "...",
  "objects": [
    {{
      "label": "person|chair|table|screen|laptop|door|wall",
      "label": "person|chair|table|laptop|door|wall",
      "bbox_2d": [x1, y1, x2, y2],
      "center_2d": [cx, cy]
    }}
  ],
  "projection_wall": {{
    "found": true,
    "bbox_2d": [x1, y1, x2, y2]
  }},
  "projection_surface": {{
    "found": true,
    "surface_type": "wall or drone_screen",
    "bbox_2d": [x1, y1, x2, y2],
    "is_free": true,
    "suitability": 0.0,
    "reason": "short reason"
  }}
}}
""".strip()


@lru_cache(maxsize=1)
def get_model_bundle() -> tuple[AutoProcessor, Any]:
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
    model = VisionModelClass.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return processor, model


def _extract_json_blob(raw_text: str) -> dict[str, Any]:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Model output does not contain JSON: {raw_text}")
    return json.loads(raw_text[start : end + 1])


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
    if inter == 0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / max(union, 1)


def _object_is_reasonable(label: str, bbox: list[int], width: int, height: int) -> bool:
    area_ratio = _bbox_area(bbox) / float(max(width * height, 1))
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    aspect_ratio = max(bbox_w / max(bbox_h, 1), bbox_h / max(bbox_w, 1))

    if area_ratio < 0.0015:
        return False
    if label != "wall" and area_ratio > 0.65:
        return False
    if label == "laptop" and area_ratio > 0.35:
        return False
    if label == "person" and bbox_h < height * 0.12:
        return False
    if label not in {"door", "wall"} and aspect_ratio > 6.0:
        return False
    return True


def _dedupe_objects(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for obj in sorted(objects, key=lambda item: _bbox_area(item["bbox_2d"]), reverse=True):
        duplicate = False
        for existing in kept:
            if obj["label"] == existing["label"] and _bbox_iou(obj["bbox_2d"], existing["bbox_2d"]) > 0.6:
                duplicate = True
                break
        if not duplicate:
            kept.append(obj)
    return kept


def _prepare_image(image: Image.Image) -> Image.Image:
    enhanced = ImageOps.autocontrast(image)
    enhanced = enhanced.filter(ImageFilter.SHARPEN)
    return enhanced


def _normalize_result(payload: dict[str, Any], image_width: int, image_height: int) -> VisionResult:
    cleaned_objects: list[dict[str, Any]] = []
    for item in payload.get("objects", []):
        label = str(item.get("label", "")).strip().lower()
        if label not in ALLOWED_LABELS:
            continue
        clipped_bbox = _clip_bbox(item.get("bbox_2d", []), image_width, image_height)
        if clipped_bbox is None or not _object_is_reasonable(label, clipped_bbox, image_width, image_height):
            continue
        cleaned_objects.append(
            {
                "label": label,
                "bbox_2d": clipped_bbox,
                "center_2d": bbox_center(clipped_bbox),
            }
        )

    cleaned_objects = _dedupe_objects(cleaned_objects)

    wall_payload = payload.get("projection_wall", {})
    wall = ProjectionWall(found=False, bbox_2d=[], center_2d=None)
    clipped_wall = _clip_bbox(wall_payload.get("bbox_2d", []), image_width, image_height)
    if wall_payload.get("found") and clipped_wall is not None and _bbox_area(clipped_wall) > 0:
        wall = ProjectionWall(found=True, bbox_2d=clipped_wall, center_2d=bbox_center(clipped_wall))

    surface_payload = payload.get("projection_surface", {})
    projection_surface = ProjectionSurface(found=False)
    clipped_surface = _clip_bbox(surface_payload.get("bbox_2d", []), image_width, image_height)
    surface_type = str(surface_payload.get("surface_type", "unknown")).strip().lower()
    if surface_type not in {"wall", "drone_screen"}:
        surface_type = "unknown"
    if surface_payload.get("found") and surface_type == "drone_screen":
        projection_surface = ProjectionSurface(
            found=True,
            surface_type="drone_screen",
            bbox_2d=[],
            center_2d=None,
            is_free=True,
            suitability=float(surface_payload.get("suitability", 1.0)),
            reason=str(surface_payload.get("reason", "Project on the drone-mounted screen.")),
        )
    elif surface_payload.get("found") and clipped_surface is not None and _bbox_area(clipped_surface) > 0:
        projection_surface = ProjectionSurface(
            found=True,
            surface_type=surface_type,
            bbox_2d=clipped_surface,
            center_2d=bbox_center(clipped_surface),
            is_free=bool(surface_payload.get("is_free", False)),
            suitability=float(surface_payload.get("suitability", 0.0)),
            reason=str(surface_payload.get("reason", "")),
        )
    elif wall.found:
        projection_surface = ProjectionSurface(
            found=True,
            surface_type="wall",
            bbox_2d=wall.bbox_2d,
            center_2d=wall.center_2d,
            is_free=False,
            suitability=None,
            reason="Fallback from projection_wall.",
        )

    return VisionResult(
        scene_description=str(payload.get("scene_description", "")),
        objects=[DetectionObject.model_validate(item) for item in cleaned_objects],
        projection_wall=wall,
        projection_surface=projection_surface,
        image_shape=[image_height, image_width],
    )


def run_qwen(
    image: Image.Image,
    preferred_label: str | None = None,
    task_prompt: str | None = None,
    requested_labels: list[str] | None = None,
    projection_target: str | None = None,
) -> VisionResult:
    processor, model = get_model_bundle()
    prepped_image = _prepare_image(image)
    prompt = _build_prompt(
        prepped_image.width,
        prepped_image.height,
        preferred_label,
        task_prompt,
        requested_labels,
        projection_target,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=[prepped_image],
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    input_token_count = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    generated_tokens = outputs[:, input_token_count:] if input_token_count else outputs
    raw_text = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    payload = _extract_json_blob(raw_text)
    result = _normalize_result(payload, prepped_image.width, prepped_image.height)
    result.raw_text = raw_text
    return result
