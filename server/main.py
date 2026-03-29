from __future__ import annotations

import json
import logging
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from common.geometry import choose_target
from common.schemas import CameraIntrinsics, DetectionObject, DronePose, InferenceResponse, ProjectionWall, VisionResult
from server.dino_detector import DINO_MODEL_ID, run_dino
from server.qwen_inference import MODEL_ID, run_qwen
from server.task_parser import parse_task_prompt

app = FastAPI(title="HoverAI Qwen2.5-VL MVP")
logger = logging.getLogger(__name__)


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
) -> list[DetectionObject]:
    merged_objects = list(dino_objects)
    seen = {(obj.label, tuple(obj.bbox_2d)) for obj in merged_objects}

    for obj in qwen_objects:
        key = (obj.label, tuple(obj.bbox_2d))
        if key in seen:
            continue
        merged_objects.append(obj.model_copy(update={"source": obj.source or "qwen"}))
        seen.add(key)

    return merged_objects


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
) -> InferenceResponse:
    image_bytes = await image.read()
    depth_bytes = await depth.read() if depth is not None else None

    frame_rgb, pil_image = _decode_color_image(image_bytes)
    depth_map = _decode_depth(depth_bytes)
    camera_intrinsics = _parse_json_form(intrinsics, CameraIntrinsics)
    pose = _parse_json_form(drone_pose, DronePose)
    parsed_task = parse_task_prompt(task_prompt)

    effective_preferred_label = preferred_label
    if effective_preferred_label is None and len(parsed_task.labels) == 1:
        effective_preferred_label = parsed_task.labels[0]

    effective_prefer_wall = prefer_wall or parsed_task.projection_target == "wall"

    dino_objects: list[DetectionObject] = []
    if parsed_task.projection_target != "drone_screen":
        try:
            dino_objects = run_dino(
                pil_image,
                preferred_label=effective_preferred_label if effective_preferred_label != "wall" else None,
                requested_labels=parsed_task.labels,
            )
        except Exception as exc:
            logger.exception("DINO inference failed, continuing without DINO objects: %s", exc)
    try:
        qwen_vision = run_qwen(
            pil_image,
            preferred_label=effective_preferred_label,
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

    merged_objects = _merge_detected_objects(dino_objects, qwen_vision.objects)

    vision = VisionResult(
        scene_description=qwen_vision.scene_description,
        objects=merged_objects,
        projection_wall=qwen_vision.projection_wall,
        projection_surface=qwen_vision.projection_surface,
        image_shape=qwen_vision.image_shape or [frame_rgb.shape[0], frame_rgb.shape[1]],
        raw_text=qwen_vision.raw_text,
    )

    if parsed_task.projection_target == "drone_screen" and not vision.projection_surface.found:
        vision.projection_surface.found = True
        vision.projection_surface.surface_type = "drone_screen"
        vision.projection_surface.is_free = True
        vision.projection_surface.reason = "Projection uses the drone-mounted screen."
        vision.projection_surface.suitability = 1.0

    target = choose_target(
        objects=vision.objects,
        wall=vision.projection_wall,
        image_shape=frame_rgb.shape[:2],
        preferred_label="drone_screen" if parsed_task.projection_target == "drone_screen" else effective_preferred_label,
        prefer_wall=effective_prefer_wall,
        drone_pose=pose,
        depth_map=depth_map,
        intrinsics=camera_intrinsics,
    )

    return InferenceResponse(vision=vision, target=target)
