from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from common.schemas import InferenceResponse, TargetCommand


@dataclass
class TemporalTrackingState:
    last_qwen_vision: dict | None = None
    last_target: TargetCommand | None = None
    last_target_ts: float | None = None
    candidate_target: TargetCommand | None = None
    candidate_count: int = 0
    miss_count: int = 0
    jump_candidate: TargetCommand | None = None
    jump_count: int = 0
    last_effective_label: str | None = None
    last_effective_prompt: str | None = None
    last_effective_prefer_wall: bool | None = None


def pixel_distance(a: list[int], b: list[int]) -> float | None:
    if len(a) != 2 or len(b) != 2:
        return None
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def blend_vectors(previous: list[float] | None, current: list[float] | None, alpha: float) -> list[float] | None:
    if previous is None or current is None or len(previous) != len(current):
        return current
    return [float((1.0 - alpha) * prev + alpha * cur) for prev, cur in zip(previous, current)]


def smooth_target_geometry(current_target: TargetCommand | None, last_target: TargetCommand | None, alpha: float):
    if current_target is None or last_target is None:
        return current_target

    alpha = min(max(alpha, 0.0), 1.0)
    smoothed_target = current_target.model_copy(deep=True)
    if len(current_target.pixel_center) == 2 and len(last_target.pixel_center) == 2:
        smoothed_target.pixel_center = [
            int(round((1.0 - alpha) * last_target.pixel_center[0] + alpha * current_target.pixel_center[0])),
            int(round((1.0 - alpha) * last_target.pixel_center[1] + alpha * current_target.pixel_center[1])),
        ]
    smoothed_target.relative_camera_vector = blend_vectors(last_target.relative_camera_vector, current_target.relative_camera_vector, alpha)
    smoothed_target.relative_world_vector = blend_vectors(last_target.relative_world_vector, current_target.relative_world_vector, alpha)
    smoothed_target.world_target = blend_vectors(last_target.world_target, current_target.world_target, alpha)
    if last_target.yaw_command is not None and current_target.yaw_command is not None:
        smoothed_target.yaw_command = float((1.0 - alpha) * last_target.yaw_command + alpha * current_target.yaw_command)
    smoothed_target.meta = dict(smoothed_target.meta)
    smoothed_target.meta["smoothed"] = True
    return smoothed_target


def reset_tracking_state(
    tracking_state: TemporalTrackingState,
    preferred_label: str | None,
    task_prompt: str | None,
    prefer_wall: bool,
) -> None:
    tracking_state.last_qwen_vision = None
    tracking_state.last_target = None
    tracking_state.last_target_ts = None
    tracking_state.candidate_target = None
    tracking_state.candidate_count = 0
    tracking_state.miss_count = 0
    tracking_state.jump_candidate = None
    tracking_state.jump_count = 0
    tracking_state.last_effective_label = preferred_label
    tracking_state.last_effective_prompt = task_prompt
    tracking_state.last_effective_prefer_wall = prefer_wall


def needs_tracking_reset(
    tracking_state: TemporalTrackingState,
    preferred_label: str | None,
    task_prompt: str | None,
    prefer_wall: bool,
) -> bool:
    return (
        preferred_label != tracking_state.last_effective_label
        or task_prompt != tracking_state.last_effective_prompt
        or prefer_wall != tracking_state.last_effective_prefer_wall
    )


def reuse_target_if_recent(
    response: InferenceResponse,
    *,
    last_target: TargetCommand | None,
    last_target_ts: float | None,
    requested_target_label: str | None,
    hold_sec: float,
) -> tuple[InferenceResponse, bool, bool]:
    if last_target is None or last_target_ts is None:
        return response, False, False

    age_sec = time.time() - last_target_ts
    if age_sec > hold_sec:
        return response, False, False

    current_target = response.target
    should_reuse = current_target is None
    missed_detection = current_target is None
    if (
        not should_reuse
        and requested_target_label is not None
        and last_target.label == requested_target_label
        and current_target.label != requested_target_label
    ):
        should_reuse = True
    if (
        not should_reuse
        and current_target is not None
        and last_target.label == current_target.label
        and current_target.type != last_target.type
    ):
        should_reuse = True
    if (
        not should_reuse
        and current_target is not None
        and requested_target_label is not None
        and last_target.label == requested_target_label
        and current_target.label == requested_target_label
        and current_target.source == "yaw_only"
        and last_target.source == "depth"
    ):
        should_reuse = True

    if not should_reuse:
        return response, False, missed_detection

    reused_target = last_target.model_copy(deep=True)
    reused_target.meta = dict(reused_target.meta)
    reused_target.meta["temporal_hold"] = True
    reused_target.meta["temporal_age_sec"] = round(age_sec, 3)
    response.target = reused_target
    return response, True, missed_detection


def stabilize_target_switch(
    response: InferenceResponse,
    *,
    requested_target_label: str | None,
    last_target: TargetCommand | None,
    candidate_target: TargetCommand | None,
    candidate_count: int,
    switch_confirm: int,
) -> tuple[InferenceResponse, TargetCommand | None, int, bool]:
    current_target = response.target
    if last_target is None or current_target is None:
        next_candidate = current_target.model_copy(deep=True) if current_target is not None else None
        next_count = 1 if current_target is not None else 0
        return response, next_candidate, next_count, False

    if requested_target_label is not None and last_target.label == requested_target_label and current_target.label != requested_target_label:
        reused_target = last_target.model_copy(deep=True)
        reused_target.meta = dict(reused_target.meta)
        reused_target.meta["switch_blocked"] = True
        response.target = reused_target
        return response, None, 0, True

    if current_target.label == last_target.label and current_target.type == last_target.type:
        return response, None, 0, False

    if candidate_target is not None and candidate_target.label == current_target.label and candidate_target.type == current_target.type:
        candidate_count += 1
    else:
        candidate_target = current_target.model_copy(deep=True)
        candidate_count = 1

    if candidate_count < max(switch_confirm, 1):
        reused_target = last_target.model_copy(deep=True)
        reused_target.meta = dict(reused_target.meta)
        reused_target.meta["switch_blocked"] = True
        reused_target.meta["switch_candidate"] = current_target.label
        reused_target.meta["switch_count"] = candidate_count
        response.target = reused_target
        return response, candidate_target, candidate_count, True

    return response, None, 0, False


def hold_target_until_drop_threshold(
    response: InferenceResponse,
    *,
    last_target: TargetCommand | None,
    requested_target_label: str | None,
    miss_count: int,
    drop_misses: int,
) -> tuple[InferenceResponse, int, bool]:
    current_target = response.target
    if last_target is None:
        return response, miss_count, False

    same_requested_label = requested_target_label is not None and last_target.label == requested_target_label
    target_missing = current_target is None
    target_mismatch = current_target is not None and same_requested_label and current_target.label != requested_target_label

    if target_missing or target_mismatch:
        miss_count += 1
    else:
        miss_count = 0

    if miss_count < max(drop_misses, 1) and (target_missing or target_mismatch):
        reused_target = last_target.model_copy(deep=True)
        reused_target.meta = dict(reused_target.meta)
        reused_target.meta["drop_hold"] = True
        reused_target.meta["drop_miss_count"] = miss_count
        response.target = reused_target
        return response, miss_count, True

    return response, miss_count, False


def stabilize_target_motion(
    response: InferenceResponse,
    *,
    last_target: TargetCommand | None,
    jump_candidate: TargetCommand | None,
    jump_count: int,
    max_jump_px: float,
    switch_confirm: int,
    smooth_alpha: float,
) -> tuple[InferenceResponse, TargetCommand | None, int, bool, bool]:
    current_target = response.target
    if last_target is None or current_target is None:
        return response, jump_candidate, jump_count, False, False

    if current_target.label != last_target.label or current_target.type != last_target.type:
        return response, None, 0, False, False

    distance = pixel_distance(last_target.pixel_center, current_target.pixel_center)
    if distance is None:
        return response, None, 0, False, False

    if distance <= max(max_jump_px, 1.0):
        response.target = smooth_target_geometry(current_target, last_target, smooth_alpha)
        return response, None, 0, False, True

    if (
        jump_candidate is not None
        and jump_candidate.label == current_target.label
        and jump_candidate.type == current_target.type
        and pixel_distance(jump_candidate.pixel_center, current_target.pixel_center) is not None
        and pixel_distance(jump_candidate.pixel_center, current_target.pixel_center) <= max(max_jump_px, 1.0)
    ):
        jump_count += 1
    else:
        jump_candidate = current_target.model_copy(deep=True)
        jump_count = 1

    if jump_count < max(switch_confirm, 1):
        reused_target = last_target.model_copy(deep=True)
        reused_target.meta = dict(reused_target.meta)
        reused_target.meta["motion_blocked"] = True
        reused_target.meta["jump_px"] = round(distance, 2)
        reused_target.meta["jump_confirm"] = jump_count
        response.target = reused_target
        return response, jump_candidate, jump_count, True, False

    response.target = smooth_target_geometry(current_target, last_target, smooth_alpha)
    return response, None, 0, False, True
