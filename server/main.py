from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from common.geometry import choose_target
from common.schemas import CameraIntrinsics, DetectionObject, DronePose, InferenceResponse, ProjectionWall, VisionResult
from common.visualization import draw_vision_result
from server.dino_detector import DINO_MODEL_ID, run_dino
from server.qwen_inference import MODEL_ID, run_qwen
from server.task_parser import parse_task_prompt
from server.wall_segmenter import segment_wall_from_depth

app = FastAPI(title="HoverAI Qwen2.5-VL MVP")
logger = logging.getLogger(__name__)
SERVER_PREVIEW_ENABLED = os.getenv("HOVERAI_SERVER_PREVIEW", "0").lower() in {"1", "true", "yes", "on"}
SERVER_LOG_JSON_ENABLED = os.getenv("HOVERAI_SERVER_LOG_JSON", "1").lower() in {"1", "true", "yes", "on"}
SERVER_PREVIEW_WINDOW = os.getenv("HOVERAI_SERVER_PREVIEW_WINDOW", "HoverAI Server Preview")
_preview_lock = threading.Lock()
STRICT_DINO_TARGET_LABELS = {"cube", "ball", "headphones", "plant", "book", "cup"}


def _bbox_area(bbox: list[int]) -> int:
    if len(bbox) != 4:
        return 0
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _bbox_intersection(a: list[int], b: list[int]) -> int:
    if len(a) != 4 or len(b) != 4:
        return 0
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def _bbox_iou(a: list[int], b: list[int]) -> float:
    intersection = _bbox_intersection(a, b)
    if intersection <= 0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - intersection
    return intersection / float(max(union, 1))


def _touches_frame_border(bbox: list[int], image_shape: tuple[int, int], margin: int = 2) -> bool:
    if len(bbox) != 4:
        return False
    image_h, image_w = image_shape
    return bbox[0] <= margin or bbox[1] <= margin or bbox[2] >= image_w - 1 - margin or bbox[3] >= image_h - 1 - margin


def _wall_visual_quality(frame_rgb: np.ndarray, bbox: list[int]) -> tuple[float, float]:
    if len(bbox) != 4:
        return 1.0, 1.0
    x1, y1, x2, y2 = bbox
    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return 1.0, 1.0

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    edge_density = float(np.count_nonzero(edges)) / float(max(edges.size, 1))

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    saturation_std = float(np.std(hsv[..., 1])) / 255.0
    return edge_density, saturation_std


def _clear_projection_surface(vision: VisionResult, reason: str) -> None:
    vision.projection_wall = ProjectionWall(found=False)
    vision.projection_surface.found = False
    vision.projection_surface.surface_type = "unknown"
    vision.projection_surface.bbox_2d = []
    vision.projection_surface.center_2d = None
    vision.projection_surface.is_free = False
    vision.projection_surface.suitability = 0.0
    vision.projection_surface.reason = reason


def _reject_invalid_wall_surface(vision: VisionResult, image_shape: tuple[int, int], frame_rgb: np.ndarray) -> None:
    surface = vision.projection_surface
    if not surface.found or len(surface.bbox_2d) != 4:
        return

    if surface.surface_type != "wall":
        _clear_projection_surface(vision, "Rejected non-wall projection surface.")
        return

    wall_bbox = surface.bbox_2d
    image_area = max(image_shape[0] * image_shape[1], 1)
    wall_area = _bbox_area(wall_bbox)
    if wall_area <= 0:
        _clear_projection_surface(vision, "Rejected empty wall prediction.")
        return

    wall_area_ratio = wall_area / float(image_area)
    if wall_area_ratio < 0.10:
        _clear_projection_surface(vision, "Rejected too-small wall prediction.")
        return

    if wall_area_ratio > 0.90:
        _clear_projection_surface(vision, "Rejected oversized wall prediction without depth support.")
        return

    if not _touches_frame_border(wall_bbox, image_shape):
        _clear_projection_surface(vision, "Rejected wall prediction that does not touch frame border.")
        return

    edge_density, saturation_std = _wall_visual_quality(frame_rgb, wall_bbox)
    if edge_density > 0.08:
        _clear_projection_surface(vision, "Rejected visually cluttered wall prediction.")
        return

    if saturation_std > 0.22:
        _clear_projection_surface(vision, "Rejected colorful wall prediction.")
        return

    overlap_ratio = 0.0
    for obj in vision.objects:
        if len(obj.bbox_2d) != 4:
            continue
        overlap_ratio += _bbox_intersection(wall_bbox, obj.bbox_2d) / float(wall_area)

    if overlap_ratio > 0.25:
        _clear_projection_surface(vision, "Rejected wall prediction that overlaps detected objects.")
        return

    vision.projection_wall = ProjectionWall(
        found=True,
        bbox_2d=wall_bbox,
        center_2d=surface.center_2d,
    )


def _decode_color_image(image_bytes: bytes) -> tuple[np.ndarray, Image.Image]:
    np_img = np.frombuffer(image_bytes, np.uint8)
    frame_bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise HTTPException(status_code=400, detail="Failed to decode input image.")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb, Image.fromarray(frame_rgb)


def _decode_depth(depth_bytes: bytes | None) -> np.ndarray | None:
    if not depth_bytes:
        return None

    np_img = np.frombuffer(depth_bytes, np.uint8)
    depth = cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise HTTPException(status_code=400, detail="Failed to decode depth image.")
    return depth.astype(np.float32)


def _parse_json_form(data: str | None, model_cls: type[Any]) -> Any | None:
    if not data:
        return None
    try:
        return model_cls.model_validate(json.loads(data))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON form field: {exc}") from exc


def _merge_detected_objects(
    dino_objects: list[DetectionObject],
    qwen_objects: list[DetectionObject],
    target_label: str | None,
) -> list[DetectionObject]:
    merged_objects = list(dino_objects)

    for obj in qwen_objects:
        is_target_object = target_label is not None and obj.label == target_label
        qwen_confirmed_by_dino = any(
            dino_obj.label == obj.label and _bbox_iou(dino_obj.bbox_2d, obj.bbox_2d) >= 0.25
            for dino_obj in dino_objects
        )

        if is_target_object and target_label in STRICT_DINO_TARGET_LABELS and not qwen_confirmed_by_dino:
            continue

        if not is_target_object and not qwen_confirmed_by_dino:
            continue

        duplicate_existing = any(
            existing.label == obj.label and _bbox_iou(existing.bbox_2d, obj.bbox_2d) >= 0.55
            for existing in merged_objects
        )
        if duplicate_existing:
            continue
        merged_objects.append(obj.model_copy(update={"source": obj.source or "qwen"}))

    return merged_objects


def _log_inference_result(response: InferenceResponse, task_prompt: str | None) -> None:
    if not SERVER_LOG_JSON_ENABLED:
        return

    payload = response.model_dump(exclude={"vision": {"raw_text"}})
    if task_prompt:
        payload["task_prompt"] = task_prompt
    logger.info("inference_result=%s", json.dumps(payload, ensure_ascii=False))


def _grounding_labels_from_task(parsed_task_labels: list[str]) -> list[str]:
    return [label for label in parsed_task_labels if label != "wall"]


def _show_server_preview(frame_rgb: np.ndarray, response: InferenceResponse, task_prompt: str | None) -> None:
    if not SERVER_PREVIEW_ENABLED:
        return

    payload = response.model_dump(exclude={"vision": {"raw_text"}})
    if task_prompt:
        payload["task_prompt"] = task_prompt

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    rendered = draw_vision_result(frame_bgr, payload)
    with _preview_lock:
        cv2.imshow(SERVER_PREVIEW_WINDOW, rendered)
        cv2.waitKey(1)


def run_inference_pipeline(
    frame_rgb: np.ndarray,
    *,
    depth_map: np.ndarray | None = None,
    camera_intrinsics: CameraIntrinsics | None = None,
    pose: DronePose | None = None,
    preferred_label: str | None = None,
    prefer_wall: bool = False,
    task_prompt: str | None = None,
    enable_dino: bool = True,
    enable_qwen: bool = True,
    emit_runtime_artifacts: bool = True,
) -> InferenceResponse:
    pil_image = Image.fromarray(frame_rgb)
    parsed_task = parse_task_prompt(task_prompt)

    effective_target_label = preferred_label or parsed_task.target_label
    if effective_target_label is None and len(parsed_task.labels) == 1:
        effective_target_label = parsed_task.labels[0]

    grounding_labels = _grounding_labels_from_task(parsed_task.labels)

    effective_prefer_wall = prefer_wall or effective_target_label == "wall"
    if parsed_task.projection_target == "wall" and effective_target_label is None:
        effective_prefer_wall = True

    dino_objects: list[DetectionObject] = []
    if enable_dino:
        try:
            dino_objects = run_dino(
                pil_image,
                preferred_label=effective_target_label if effective_target_label not in {None, "wall"} else None,
                requested_labels=grounding_labels,
            )
        except Exception as exc:
            logger.exception("DINO inference failed, continuing without DINO objects: %s", exc)

    qwen_vision = VisionResult(
        scene_description="",
        objects=[],
        projection_wall=ProjectionWall(found=False),
        image_shape=[frame_rgb.shape[0], frame_rgb.shape[1]],
    )
    if enable_qwen:
        try:
            qwen_vision = run_qwen(
                pil_image,
                preferred_label=effective_target_label,
                target_label=effective_target_label,
                task_prompt=task_prompt,
                requested_labels=parsed_task.labels,
                projection_target=parsed_task.projection_target,
            )
        except Exception as exc:
            logger.exception("Qwen inference failed, falling back to DINO-only objects: %s", exc)
            qwen_vision = VisionResult(
                scene_description="",
                objects=[],
                projection_wall=ProjectionWall(found=False),
                image_shape=[frame_rgb.shape[0], frame_rgb.shape[1]],
                raw_text=str(exc),
            )

    merged_objects = _merge_detected_objects(dino_objects, qwen_vision.objects, effective_target_label)

    vision = VisionResult(
        scene_description=qwen_vision.scene_description,
        objects=merged_objects,
        projection_wall=qwen_vision.projection_wall,
        projection_surface=qwen_vision.projection_surface,
        image_shape=qwen_vision.image_shape or [frame_rgb.shape[0], frame_rgb.shape[1]],
        raw_text=qwen_vision.raw_text,
    )

    segmented_wall = None
    if depth_map is not None:
        try:
            segmented_wall = segment_wall_from_depth(depth_map, merged_objects)
        except Exception as exc:
            logger.warning("Wall segmentation failed: %s", exc)

    if segmented_wall is not None:
        segmented_projection_wall, segmented_projection_surface = segmented_wall
        vision.projection_wall = segmented_projection_wall
        vision.projection_surface = segmented_projection_surface
    elif vision.projection_surface.found:
        _reject_invalid_wall_surface(vision, frame_rgb.shape[:2], frame_rgb)

    if parsed_task.projection_target == "drone_screen" and not vision.projection_surface.found:
        vision.projection_surface.found = True
        vision.projection_surface.surface_type = "drone_screen"
        vision.projection_surface.is_free = True
        vision.projection_surface.reason = "Projection uses the drone-mounted screen."
        vision.projection_surface.suitability = 1.0

    target_preferred_label = effective_target_label
    if target_preferred_label is None and parsed_task.projection_target == "drone_screen":
        target_preferred_label = "drone_screen"

    target = choose_target(
        objects=vision.objects,
        wall=vision.projection_wall,
        image_shape=frame_rgb.shape[:2],
        preferred_label=target_preferred_label,
        prefer_wall=effective_prefer_wall,
        drone_pose=pose,
        depth_map=depth_map,
        intrinsics=camera_intrinsics,
    )

    response = InferenceResponse(vision=vision, target=target)
    if emit_runtime_artifacts:
        _log_inference_result(response, task_prompt)
        try:
            _show_server_preview(frame_rgb, response, task_prompt)
        except Exception as exc:
            logger.warning("Server preview update failed: %s", exc)
    return response


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "model_id": MODEL_ID, "dino_model_id": DINO_MODEL_ID}


@app.post("/infer", response_model=InferenceResponse)
async def infer(
    image: UploadFile = File(...),
    depth: UploadFile | None = File(default=None),
    intrinsics: str | None = Form(default=None),
    drone_pose: str | None = Form(default=None),
    preferred_label: str | None = Form(default=None),
    prefer_wall: bool = Form(default=False),
    task_prompt: str | None = Form(default=None),
    depth_scale: float = Form(default=0.001),
) -> InferenceResponse:
    image_bytes = await image.read()
    depth_bytes = await depth.read() if depth is not None else None

    frame_rgb, _ = _decode_color_image(image_bytes)
    depth_map = _decode_depth(depth_bytes)
    if depth_map is not None:
        depth_map = depth_map * depth_scale
    camera_intrinsics = _parse_json_form(intrinsics, CameraIntrinsics)
    pose = _parse_json_form(drone_pose, DronePose)

    return run_inference_pipeline(
        frame_rgb,
        depth_map=depth_map,
        camera_intrinsics=camera_intrinsics,
        pose=pose,
        preferred_label=preferred_label,
        prefer_wall=prefer_wall,
        task_prompt=task_prompt,
    )
