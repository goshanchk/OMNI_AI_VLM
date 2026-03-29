from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from common.schemas import CameraIntrinsics, DetectionObject, DronePose, ProjectionWall, TargetCommand


def bbox_center(bbox: Iterable[int]) -> list[int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return [int((x1 + x2) / 2), int((y1 + y2) / 2)]


def pixel_to_3d(
    px: int,
    py: int,
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[float, float, float]:
    py = int(np.clip(py, 0, depth_map.shape[0] - 1))
    px = int(np.clip(px, 0, depth_map.shape[1] - 1))
    z = float(depth_map[py, px])
    x = (px - intrinsics.cx0) * z / intrinsics.fx
    y = (py - intrinsics.cy0) * z / intrinsics.fy
    return x, y, z


def yaw_from_pixel(px: int, image_width: int) -> float:
    image_center_x = image_width / 2.0
    return float((px - image_center_x) / max(image_width, 1))


def camera_vector_to_world(
    camera_vector: tuple[float, float, float],
    drone_pose: DronePose,
) -> tuple[float, float, float]:
    x_cam, y_cam, z_cam = camera_vector
    cos_yaw = math.cos(drone_pose.yaw)
    sin_yaw = math.sin(drone_pose.yaw)

    x_world = cos_yaw * x_cam - sin_yaw * y_cam
    y_world = sin_yaw * x_cam + cos_yaw * y_cam
    z_world = z_cam
    return x_world, y_world, z_world


def world_target_from_pose(
    drone_pose: DronePose,
    relative_world_vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        drone_pose.x + relative_world_vector[0],
        drone_pose.y + relative_world_vector[1],
        drone_pose.z + relative_world_vector[2],
    )


def build_target_command(
    *,
    label: str,
    center_xy: list[int],
    image_shape: tuple[int, int],
    drone_pose: DronePose | None,
    depth_map: np.ndarray | None,
    intrinsics: CameraIntrinsics | None,
    source: str,
) -> TargetCommand:
    px, py = center_xy
    image_h, image_w = image_shape

    if depth_map is not None and intrinsics is not None:
        rel_cam = pixel_to_3d(px, py, depth_map, intrinsics)
        if rel_cam[2] > 0.0:
            rel_world = camera_vector_to_world(rel_cam, drone_pose) if drone_pose else None
            world_target = world_target_from_pose(drone_pose, rel_world) if drone_pose and rel_world else None
            return TargetCommand(
                type="wall" if label == "wall" else "object",
                label=label,
                pixel_center=[px, py],
                relative_camera_vector=list(rel_cam),
                relative_world_vector=list(rel_world) if rel_world else None,
                world_target=list(world_target) if world_target else None,
                source="depth" if source != "wall_bbox" else "wall_bbox",
                meta={"image_shape": [image_h, image_w]},
            )

    return TargetCommand(
        type="yaw_only",
        label=label,
        pixel_center=[px, py],
        yaw_command=yaw_from_pixel(px, image_w),
        source="yaw_only",
        meta={"image_shape": [image_h, image_w]},
    )


def choose_target(
    *,
    objects: list[DetectionObject],
    wall: ProjectionWall,
    image_shape: tuple[int, int],
    preferred_label: str | None,
    prefer_wall: bool,
    drone_pose: DronePose | None,
    depth_map: np.ndarray | None,
    intrinsics: CameraIntrinsics | None,
) -> TargetCommand | None:
    if preferred_label == "drone_screen":
        return TargetCommand(
            type="drone_screen",
            label="drone_screen",
            pixel_center=[0, 0],
            yaw_command=0.0,
            source="yaw_only",
            meta={"projection_surface": "drone_screen"},
        )

    if prefer_wall and wall.found and len(wall.bbox_2d) == 4:
        center = wall.center_2d or bbox_center(wall.bbox_2d)
        return build_target_command(
            label="wall",
            center_xy=center,
            image_shape=image_shape,
            drone_pose=drone_pose,
            depth_map=depth_map,
            intrinsics=intrinsics,
            source="wall_bbox",
        )

    if preferred_label:
        for obj in objects:
            if obj.label == preferred_label:
                return build_target_command(
                    label=obj.label,
                    center_xy=obj.center_2d,
                    image_shape=image_shape,
                    drone_pose=drone_pose,
                    depth_map=depth_map,
                    intrinsics=intrinsics,
                    source="depth",
                )

    if objects:
        return build_target_command(
            label=objects[0].label,
            center_xy=objects[0].center_2d,
            image_shape=image_shape,
            drone_pose=drone_pose,
            depth_map=depth_map,
            intrinsics=intrinsics,
            source="depth",
        )

    if wall.found and len(wall.bbox_2d) == 4:
        center = wall.center_2d or bbox_center(wall.bbox_2d)
        return build_target_command(
            label="wall",
            center_xy=center,
            image_shape=image_shape,
            drone_pose=drone_pose,
            depth_map=depth_map,
            intrinsics=intrinsics,
            source="wall_bbox",
        )

    return None
