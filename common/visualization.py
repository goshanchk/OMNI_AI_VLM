from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.18
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad_x = 1
    pad_y = 1
    top = max(y - text_h - baseline - pad_y * 2, 0)
    bottom = max(y, text_h + baseline + pad_y * 2)
    right = x + text_w + pad_x * 2
    cv2.rectangle(frame, (x, top), (right, bottom), color, -1)
    cv2.putText(frame, text, (x + pad_x, bottom - baseline - pad_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _scale_point(point: list[int], scale_x: float, scale_y: float) -> tuple[int, int] | None:
    if len(point) != 2:
        return None
    return int(point[0] * scale_x), int(point[1] * scale_y)


def _scale_bbox(bbox: list[int], scale_x: float, scale_y: float) -> tuple[int, int, int, int] | None:
    if len(bbox) != 4:
        return None
    return (
        int(bbox[0] * scale_x),
        int(bbox[1] * scale_y),
        int(bbox[2] * scale_x),
        int(bbox[3] * scale_y),
    )


def draw_vision_result(frame_bgr: np.ndarray, payload: dict[str, Any]) -> np.ndarray:
    rendered = frame_bgr.copy()
    vision = payload.get("vision", {})
    objects = vision.get("objects", [])
    wall = vision.get("projection_wall", {})
    projection_surface = vision.get("projection_surface", {})
    target = payload.get("target")
    infer_shape = None
    if target and isinstance(target.get("meta"), dict):
        infer_shape = target["meta"].get("image_shape")
    if not infer_shape and isinstance(vision, dict):
        infer_shape = vision.get("image_shape")

    scale_x = 1.0
    scale_y = 1.0
    if isinstance(infer_shape, list) and len(infer_shape) == 2 and infer_shape[0] and infer_shape[1]:
        scale_y = frame_bgr.shape[0] / float(infer_shape[0])
        scale_x = frame_bgr.shape[1] / float(infer_shape[1])

    for obj in objects:
        bbox = obj.get("bbox_2d", [])
        center = obj.get("center_2d", [])
        label = obj.get("label", "object")
        confidence = obj.get("confidence")
        source = obj.get("source")
        label_text = label
        if confidence is not None:
            label_text += f" {confidence:.2f}"
        if source:
            label_text += f" [{source}]"
        scaled_bbox = _scale_bbox(bbox, scale_x, scale_y)
        scaled_center = _scale_point(center, scale_x, scale_y)
        if scaled_bbox is not None:
            x1, y1, x2, y2 = scaled_bbox
            cv2.rectangle(rendered, (x1, y1), (x2, y2), (60, 220, 60), 1, cv2.LINE_AA)
            _draw_label(rendered, label_text, x1, y1, (60, 220, 60))
        if scaled_center is not None:
            cv2.circle(rendered, scaled_center, 3, (60, 220, 60), -1, cv2.LINE_AA)

    scaled_wall_bbox = _scale_bbox(wall.get("bbox_2d", []), scale_x, scale_y)
    if wall.get("found") and scaled_wall_bbox is not None:
        x1, y1, x2, y2 = scaled_wall_bbox
        cv2.rectangle(rendered, (x1, y1), (x2, y2), (60, 160, 255), 1, cv2.LINE_AA)
        _draw_label(rendered, "wall", x1, y1, (60, 160, 255))

    scaled_surface_bbox = _scale_bbox(projection_surface.get("bbox_2d", []), scale_x, scale_y)
    if projection_surface.get("found") and scaled_surface_bbox is not None:
        x1, y1, x2, y2 = scaled_surface_bbox
        surface_type = projection_surface.get("surface_type", "surface")
        is_free = projection_surface.get("is_free", False)
        suitability = projection_surface.get("suitability")
        label = f"project:{surface_type}"
        if suitability is not None:
            label += f" {suitability:.2f}"
        if is_free:
            label += " free"
        cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 200, 255), 1, cv2.LINE_AA)
        _draw_label(rendered, label, x1, max(y1 - 20, 0), (0, 200, 255))

    if target:
        scaled_target_center = _scale_point(target.get("pixel_center", []), scale_x, scale_y)
        if scaled_target_center is not None:
            cv2.circle(rendered, scaled_target_center, 6, (0, 0, 255), 2, cv2.LINE_AA)
        target_text = f"target={target.get('label', '?')} type={target.get('type', '?')}"
        yaw = target.get("yaw_command")
        if yaw is not None:
            target_text += f" yaw={yaw:.3f}"
        cv2.putText(rendered, target_text[:40], (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.24, (0, 0, 255), 1, cv2.LINE_AA)

    scene_description = vision.get("scene_description")
    if scene_description:
        cv2.putText(rendered, scene_description[:44], (6, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.20, (255, 255, 0), 1, cv2.LINE_AA)

    task_prompt = payload.get("task_prompt")
    if task_prompt:
        cv2.putText(rendered, task_prompt[:46], (6, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.18, (180, 255, 180), 1, cv2.LINE_AA)

    surface_reason = projection_surface.get("reason")
    if surface_reason:
        cv2.putText(rendered, surface_reason[:46], (6, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.18, (150, 220, 255), 1, cv2.LINE_AA)

    return rendered
