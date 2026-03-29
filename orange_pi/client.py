from __future__ import annotations

import argparse
import json
import time

import cv2
import requests

from orange_pi.flight_bridge import FlightBridge
from orange_pi.vicon_bridge import ViconBridge


def main() -> None:
    parser = argparse.ArgumentParser(description="Orange Pi RGB client for HoverAI Qwen server")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000/infer")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--period-sec", type=float, default=1.0)
    parser.add_argument("--preferred-label", default=None)
    parser.add_argument("--prefer-wall", action="store_true")
    parser.add_argument("--depth-image", default=None, help="Optional path to a depth PNG for demo/testing")
    parser.add_argument("--use-vicon", action="store_true", help="Use ViconBridge stub to refresh pose each loop")
    parser.add_argument(
        "--drone-pose",
        default=None,
        help='JSON string like {"x":0,"y":0,"z":1.2,"yaw":0.0}',
    )
    parser.add_argument(
        "--intrinsics",
        default=None,
        help='JSON string like {"fx":525,"fy":525,"cx0":319.5,"cy0":239.5}',
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {args.camera_index}")

    flight_bridge = FlightBridge()
    vicon_bridge = ViconBridge() if args.use_vicon else None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[orange_pi] failed to read frame")
            time.sleep(args.period_sec)
            continue

        ok, img_encoded = cv2.imencode(".jpg", frame)
        if not ok:
            print("[orange_pi] failed to encode frame")
            time.sleep(args.period_sec)
            continue

        files = {"image": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
        if args.depth_image:
            with open(args.depth_image, "rb") as depth_file:
                files["depth"] = ("depth.png", depth_file.read(), "image/png")

        drone_pose = args.drone_pose
        if vicon_bridge is not None:
            drone_pose = vicon_bridge.get_pose().model_dump_json()

        data = {"prefer_wall": json.dumps(args.prefer_wall)}
        if args.preferred_label:
            data["preferred_label"] = args.preferred_label
        if drone_pose:
            data["drone_pose"] = drone_pose
        if args.intrinsics:
            data["intrinsics"] = args.intrinsics

        response = requests.post(args.server_url, files=files, data=data, timeout=120)
        response.raise_for_status()
        payload = response.json()
        print(json.dumps(payload, indent=2, ensure_ascii=False))

        target = payload.get("target")
        if target is not None:
            flight_bridge.send_obj(target)

        time.sleep(args.period_sec)


if __name__ == "__main__":
    main()
