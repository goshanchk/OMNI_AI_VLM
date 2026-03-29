from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DetectionObject(BaseModel):
    label: str
    bbox_2d: list[int] = Field(default_factory=list, min_length=4, max_length=4)
    center_2d: list[int] = Field(default_factory=list, min_length=2, max_length=2)
    confidence: float | None = None
    source: Literal["dino", "yolo", "qwen"] | None = None


class ProjectionWall(BaseModel):
    found: bool = False
    bbox_2d: list[int] = Field(default_factory=list)
    center_2d: list[int] | None = None


class ProjectionSurface(BaseModel):
    found: bool = False
    surface_type: Literal["wall", "screen", "drone_screen", "unknown"] = "unknown"
    bbox_2d: list[int] = Field(default_factory=list)
    center_2d: list[int] | None = None
    is_free: bool = False
    suitability: float | None = None
    reason: str = ""


class VisionResult(BaseModel):
    scene_description: str = ""
    objects: list[DetectionObject] = Field(default_factory=list)
    projection_wall: ProjectionWall = Field(default_factory=ProjectionWall)
    projection_surface: ProjectionSurface = Field(default_factory=ProjectionSurface)
    image_shape: list[int] | None = None
    raw_text: str | None = None


class CameraIntrinsics(BaseModel):
    fx: float
    fy: float
    cx0: float
    cy0: float


class DronePose(BaseModel):
    x: float
    y: float
    z: float
    yaw: float = 0.0


class TargetCommand(BaseModel):
    type: Literal["object", "wall", "yaw_only", "drone_screen"]
    label: str
    pixel_center: list[int] = Field(default_factory=list, min_length=2, max_length=2)
    relative_camera_vector: list[float] | None = None
    relative_world_vector: list[float] | None = None
    world_target: list[float] | None = None
    yaw_command: float | None = None
    source: Literal["depth", "yaw_only", "wall_bbox"] = "yaw_only"
    meta: dict[str, Any] = Field(default_factory=dict)


class InferenceResponse(BaseModel):
    vision: VisionResult
    target: TargetCommand | None = None
