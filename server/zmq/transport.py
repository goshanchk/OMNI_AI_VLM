from __future__ import annotations

import time

import cv2
import msgpack
import numpy as np
import zmq

from common.schemas import CameraIntrinsics
from server.zmq.state import LatestPacket, SharedState


def decode_rgb(image_jpeg: bytes) -> np.ndarray:
    np_img = np.frombuffer(image_jpeg, np.uint8)
    frame_bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError("Failed to decode RGB image from ZeroMQ payload")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def decode_depth(depth_png: bytes | None) -> np.ndarray | None:
    if not depth_png:
        return None
    np_img = np.frombuffer(depth_png, np.uint8)
    depth = cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError("Failed to decode depth image from ZeroMQ payload")
    return depth.astype(np.float32)


def recv_loop(sock: zmq.Socket, state: SharedState) -> None:
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    seq = 0

    while not state.stop:
        events = dict(poller.poll(50))
        if sock not in events:
            continue

        try:
            packed = sock.recv(copy=False)
            payload = msgpack.unpackb(bytes(packed.buffer), raw=False)
            frame_rgb = decode_rgb(payload["image_jpeg"])
            depth_map = decode_depth(payload.get("depth_png"))
            if depth_map is not None:
                depth_scale = float(payload.get("depth_scale", 0.001))
                depth_map = depth_map * depth_scale
            intrinsics = CameraIntrinsics.model_validate(payload["intrinsics"]) if payload.get("intrinsics") else None
            packet = LatestPacket(
                frame_rgb=frame_rgb,
                depth_map=depth_map,
                intrinsics=intrinsics,
                sent_ts=payload.get("sent_ts"),
                seq=seq,
            )
            with state.lock:
                state.latest_packet = packet
                state.received_count += 1
                state.latest_transport_latency_ms = (
                    (time.time() - packet.sent_ts) * 1000.0 if packet.sent_ts is not None else None
                )
                state.error_text = None
            seq += 1
        except Exception as exc:
            with state.lock:
                state.error_text = str(exc)
