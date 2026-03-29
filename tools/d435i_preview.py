from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.visualization import draw_vision_result

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover - depends on local hardware package
    raise SystemExit(
        "pyrealsense2 is not installed. Install Intel RealSense SDK Python bindings first."
    ) from exc


@dataclass
class SharedState:
    last_payload: dict | None = None
    last_infer_s: float = 0.0
    in_flight: bool = False
    request_count: int = 0
    error_text: str | None = None


def _infer_async(
    *,
    server_url: str,
    color_bgr: np.ndarray,
    depth_image: np.ndarray | None,
    intrinsics: dict[str, float],
    preferred_label: str | None,
    prefer_wall: bool,
    task_prompt: str | None,
    jpeg_quality: int,
    send_depth: bool,
    state: SharedState,
) -> None:
    try:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        ok_color, color_encoded = cv2.imencode(".jpg", color_bgr, encode_params)
        if not ok_color:
            state.error_text = "Failed to encode RGB frame"
            return

        files = {"image": ("frame.jpg", color_encoded.tobytes(), "image/jpeg")}
        if send_depth and depth_image is not None:
            ok_depth, depth_encoded = cv2.imencode(".png", depth_image)
            if not ok_depth:
                state.error_text = "Failed to encode depth frame"
                return
            files["depth"] = ("depth.png", depth_encoded.tobytes(), "image/png")

        data = {"intrinsics": json.dumps(intrinsics)}
        if preferred_label:
            data["preferred_label"] = preferred_label
        if prefer_wall:
            data["prefer_wall"] = "true"
        if task_prompt:
            data["task_prompt"] = task_prompt

        response = requests.post(server_url, files=files, data=data, timeout=120)
        response.raise_for_status()
        state.last_payload = response.json()
        if task_prompt:
            state.last_payload["task_prompt"] = task_prompt
        state.last_infer_s = time.time()
        state.request_count += 1
        state.error_text = None
    except Exception as exc:
        state.error_text = str(exc)
    finally:
        state.in_flight = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Live D435i preview for HoverAI Qwen server")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000/infer")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--infer-every", type=int, default=15, help="Run server inference every N frames")
    parser.add_argument("--infer-width", type=int, default=448, help="Resize RGB before sending to server")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality for uploaded RGB frame")
    parser.add_argument("--preferred-label", default=None)
    parser.add_argument("--prefer-wall", action="store_true")
    parser.add_argument("--task-prompt", default=None, help="Free-form task prompt with requested objects and projection target")
    parser.add_argument("--send-depth", action="store_true", help="Upload aligned depth PNG to the server")
    args = parser.parse_args()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    intrinsics = {
        "fx": float(color_intrinsics.fx),
        "fy": float(color_intrinsics.fy),
        "cx0": float(color_intrinsics.ppx),
        "cy0": float(color_intrinsics.ppy),
    }

    state = SharedState()
    frame_idx = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_bgr = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            if not state.in_flight and frame_idx % max(args.infer_every, 1) == 0:
                infer_w = max(args.infer_width, 64)
                infer_h = max(int(color_bgr.shape[0] * infer_w / color_bgr.shape[1]), 64)
                infer_color = cv2.resize(color_bgr, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
                infer_depth = None
                infer_intrinsics = intrinsics
                if args.send_depth:
                    infer_depth = cv2.resize(depth_image, (infer_w, infer_h), interpolation=cv2.INTER_NEAREST)
                    scale_x = infer_w / color_bgr.shape[1]
                    scale_y = infer_h / color_bgr.shape[0]
                    infer_intrinsics = {
                        "fx": intrinsics["fx"] * scale_x,
                        "fy": intrinsics["fy"] * scale_y,
                        "cx0": intrinsics["cx0"] * scale_x,
                        "cy0": intrinsics["cy0"] * scale_y,
                    }

                state.in_flight = True
                threading.Thread(
                    target=_infer_async,
                    kwargs={
                        "server_url": args.server_url,
                        "color_bgr": infer_color.copy(),
                        "depth_image": infer_depth.copy() if infer_depth is not None else None,
                        "intrinsics": infer_intrinsics,
                        "preferred_label": args.preferred_label,
                        "prefer_wall": args.prefer_wall,
                        "task_prompt": args.task_prompt,
                        "jpeg_quality": args.jpeg_quality,
                        "send_depth": args.send_depth,
                        "state": state,
                    },
                    daemon=True,
                ).start()

            rendered = draw_vision_result(color_bgr, state.last_payload or {})
            age_s = time.time() - state.last_infer_s if state.last_infer_s else -1.0
            status = (
                f"D435i | infer_every={args.infer_every} | in_flight={int(state.in_flight)} "
                f"| replies={state.request_count} | last={age_s:.1f}s"
            )
            cv2.putText(rendered, status, (12, rendered.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            if state.error_text:
                cv2.putText(rendered, state.error_text[:110], (12, rendered.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.imshow("HoverAI D435i Preview", rendered)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            frame_idx += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
