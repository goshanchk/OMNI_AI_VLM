from __future__ import annotations

import cv2
import numpy as np

from common.geometry import bbox_center
from common.schemas import DetectionObject, ProjectionSurface, ProjectionWall


def _bbox_area(bbox: list[int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _bbox_intersection(a: list[int], b: list[int]) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def _touches_frame_border(x: int, y: int, w: int, h: int, width: int, height: int) -> bool:
    return x <= 2 or y <= 2 or (x + w) >= width - 2 or (y + h) >= height - 2


def segment_wall_from_depth(depth_map: np.ndarray, objects: list[DetectionObject]) -> tuple[ProjectionWall, ProjectionSurface] | None:
    if depth_map.ndim != 2:
        return None

    valid = np.isfinite(depth_map) & (depth_map > 0)
    if valid.sum() < 500:
        return None

    height, width = depth_map.shape
    valid_depths = depth_map[valid]
    far_threshold = float(np.percentile(valid_depths, 75))

    depth_blur = cv2.GaussianBlur(depth_map.astype(np.float32), (5, 5), 0)
    grad_x = cv2.Sobel(depth_blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_threshold = float(np.percentile(grad_mag[valid], 60))

    mask = valid & (depth_blur >= far_threshold) & (grad_mag <= grad_threshold)
    mask = (mask.astype(np.uint8) * 255)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best_label = None
    best_score = -1.0

    for label_idx in range(1, num_labels):
        x, y, w, h, area = stats[label_idx]
        if area < int(0.08 * width * height):
            continue
        if not _touches_frame_border(x, y, w, h, width, height):
            continue
        bbox = [int(x), int(y), int(x + w), int(y + h)]
        area_ratio = area / float(width * height)
        aspect = max(w / max(h, 1), h / max(w, 1))
        if aspect > 8.0:
            continue
        score = area_ratio
        if score > best_score:
            best_score = score
            best_label = label_idx

    if best_label is None:
        return None

    x, y, w, h, area = stats[best_label]
    bbox = [int(x), int(y), int(x + w), int(y + h)]
    center = bbox_center(bbox)
    area_ratio = area / float(width * height)

    object_overlap = 0.0
    if area > 0:
        for obj in objects:
            if len(obj.bbox_2d) == 4:
                object_overlap += _bbox_intersection(bbox, obj.bbox_2d) / float(area)

    depth_values = depth_blur[labels == best_label]
    depth_std = float(np.std(depth_values)) if depth_values.size else 0.0
    depth_mean = float(np.mean(depth_values)) if depth_values.size else 0.0
    smoothness = 1.0 / (1.0 + (depth_std / max(depth_mean, 1e-6)))
    suitability = max(0.0, min(1.0, area_ratio * 1.8 * smoothness * max(0.0, 1.0 - object_overlap)))
    is_free = object_overlap < 0.12 and area_ratio > 0.12
    reason = 'Depth-based wall segmentation.'
    if not is_free:
        reason = 'Depth wall candidate overlaps detected objects.'

    wall = ProjectionWall(found=True, bbox_2d=bbox, center_2d=center)
    surface = ProjectionSurface(
        found=True,
        surface_type='wall',
        bbox_2d=bbox,
        center_2d=center,
        is_free=is_free,
        suitability=suitability,
        reason=reason,
    )
    return wall, surface
