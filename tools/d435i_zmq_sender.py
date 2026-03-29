from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import msgpack
import numpy as np
import zmq

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover - depends on local hardware package
    raise SystemExit(
        'pyrealsense2 is not installed. Install Intel RealSense SDK Python bindings first.'
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Low-latency D435i ZeroMQ sender for HoverAI')
    parser.add_argument('--server-ip', required=True)
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--infer-every', type=int, default=1, help='Send every Nth captured frame')
    parser.add_argument('--infer-width', type=int, default=224, help='Resize RGB before send')
    parser.add_argument('--jpeg-quality', type=int, default=40)
    parser.add_argument('--send-depth', action='store_true')
    parser.add_argument('--png-level', type=int, default=1, help='PNG compression level for depth')
    parser.add_argument('--show-local', action='store_true', help='Show raw local camera preview')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.SNDHWM, 1)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f'tcp://{args.server_ip}:{args.port}')

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    if args.send_depth:
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color) if args.send_depth else None

    depth_scale = 0.001
    if args.send_depth:
        depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())

    color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    intrinsics = {
        'fx': float(color_intrinsics.fx),
        'fy': float(color_intrinsics.fy),
        'cx0': float(color_intrinsics.ppx),
        'cy0': float(color_intrinsics.ppy),
    }

    frame_idx = 0
    sent_count = 0
    started_at = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            if align is not None:
                frames = align.process(frames)

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            depth_frame = frames.get_depth_frame() if args.send_depth else None
            if args.send_depth and not depth_frame:
                continue

            color_bgr = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None

            if frame_idx % max(args.infer_every, 1) != 0:
                frame_idx += 1
                continue

            infer_w = max(args.infer_width, 64)
            infer_h = max(int(color_bgr.shape[0] * infer_w / color_bgr.shape[1]), 64)
            infer_color = cv2.resize(color_bgr, (infer_w, infer_h), interpolation=cv2.INTER_AREA)

            ok_color, color_encoded = cv2.imencode(
                '.jpg',
                infer_color,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
            )
            if not ok_color:
                frame_idx += 1
                continue

            payload: dict[str, object] = {
                'image_jpeg': color_encoded.tobytes(),
                'image_shape': [int(infer_color.shape[0]), int(infer_color.shape[1])],
                'intrinsics': {
                    'fx': intrinsics['fx'] * (infer_w / color_bgr.shape[1]),
                    'fy': intrinsics['fy'] * (infer_h / color_bgr.shape[0]),
                    'cx0': intrinsics['cx0'] * (infer_w / color_bgr.shape[1]),
                    'cy0': intrinsics['cy0'] * (infer_h / color_bgr.shape[0]),
                },
                'sent_ts': time.time(),
            }

            if args.send_depth and depth_image is not None:
                infer_depth = cv2.resize(depth_image, (infer_w, infer_h), interpolation=cv2.INTER_NEAREST)
                ok_depth, depth_encoded = cv2.imencode(
                    '.png',
                    infer_depth,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), int(args.png_level)],
                )
                if ok_depth:
                    payload['depth_png'] = depth_encoded.tobytes()
                    payload['depth_scale'] = depth_scale

            try:
                sock.send(msgpack.packb(payload, use_bin_type=True), flags=zmq.NOBLOCK, copy=False)
                sent_count += 1
            except zmq.Again:
                pass

            if args.show_local:
                preview = color_bgr.copy()
                elapsed = max(time.time() - started_at, 1e-6)
                fps = sent_count / elapsed
                cv2.putText(preview, f'sent={sent_count} fps={fps:.1f}', (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('HoverAI D435i ZeroMQ Sender', preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            frame_idx += 1
    finally:
        pipeline.stop()
        if args.show_local:
            cv2.destroyAllWindows()
        sock.close(0)


if __name__ == '__main__':
    main()
