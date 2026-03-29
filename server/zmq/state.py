from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from common.schemas import CameraIntrinsics


@dataclass
class LatestPacket:
    frame_rgb: np.ndarray
    depth_map: np.ndarray | None
    intrinsics: CameraIntrinsics | None
    sent_ts: float | None
    seq: int


@dataclass
class SharedState:
    latest_packet: LatestPacket | None = None
    latest_overlay_payload: dict | None = None
    latest_infer_latency_ms: float | None = None
    latest_transport_latency_ms: float | None = None
    received_count: int = 0
    processed_count: int = 0
    error_text: str | None = None
    stop: bool = False
    lock: threading.Lock | None = None
    runtime_label: str | None = None
    runtime_prompt: str | None = None
    runtime_projection: str | None = None
    command_hint: str | None = None
