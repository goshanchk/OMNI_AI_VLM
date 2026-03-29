from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import msgpack
import numpy as np
import zmq

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.schemas import CameraIntrinsics, DronePose
from common.visualization import draw_vision_result
from server.main import run_inference_pipeline
from server.task_parser import parse_task_prompt

logger = logging.getLogger('hoverai.zmq_receiver')
WINDOW_NAME = 'HoverAI ZeroMQ Live Preview'
DISPLAY_SCALE = float(os.getenv('HOVERAI_PREVIEW_SCALE', '1.35'))


@dataclass
class LatestPacket:
    frame_rgb: np.ndarray
    depth_map: np.ndarray | None
    intrinsics: CameraIntrinsics | None
    preferred_label: str | None
    prefer_wall: bool
    task_prompt: str | None
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='HoverAI ZeroMQ receiver for low-latency inference')
    parser.add_argument('--bind', default='tcp://*:5555')
    parser.add_argument('--preferred-label', default=None)
    parser.add_argument('--prefer-wall', action='store_true')
    parser.add_argument('--task-prompt', default=None)
    parser.add_argument('--disable-dino', action='store_true')
    parser.add_argument('--disable-qwen', action='store_true')
    parser.add_argument('--qwen-every', type=int, default=3, help='Run Qwen every N processed frames and reuse last projection in between')
    parser.add_argument('--target-hold-sec', type=float, default=1.2, help='Keep the last good target briefly when detection drops out')
    parser.add_argument('--target-switch-confirm', type=int, default=2, help='Require N consecutive frames before switching to a different target')
    parser.add_argument('--target-drop-misses', type=int, default=3, help='Drop the current target only after N consecutive missed frames')
    parser.add_argument('--target-max-jump-px', type=float, default=34.0, help='Allow only small target motion between frames before requiring confirmation')
    parser.add_argument('--target-smooth-alpha', type=float, default=0.35, help='Blend factor for target smoothing, smaller is steadier')
    parser.add_argument('--drone-pose', default=None, help='JSON string like {"x":0,"y":0,"z":1.2,"yaw":0.0}')
    parser.add_argument('--no-display', action='store_true', help='Disable live server preview window')
    return parser.parse_args()


def _requested_target_label(packet: LatestPacket, args: argparse.Namespace) -> str | None:
    if args.preferred_label or packet.preferred_label:
        return args.preferred_label or packet.preferred_label
    parsed_task = parse_task_prompt(args.task_prompt or packet.task_prompt)
    return parsed_task.target_label


def _pixel_distance(a: list[int], b: list[int]) -> float | None:
    if len(a) != 2 or len(b) != 2:
        return None
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _blend_vectors(previous: list[float] | None, current: list[float] | None, alpha: float) -> list[float] | None:
    if previous is None or current is None or len(previous) != len(current):
        return current
    return [float((1.0 - alpha) * prev + alpha * cur) for prev, cur in zip(previous, current)]


def _smooth_target_geometry(current_target, last_target, alpha: float):
    if current_target is None or last_target is None:
        return current_target

    alpha = min(max(alpha, 0.0), 1.0)
    smoothed_target = current_target.model_copy(deep=True)
    if len(current_target.pixel_center) == 2 and len(last_target.pixel_center) == 2:
        smoothed_target.pixel_center = [
            int(round((1.0 - alpha) * last_target.pixel_center[0] + alpha * current_target.pixel_center[0])),
            int(round((1.0 - alpha) * last_target.pixel_center[1] + alpha * current_target.pixel_center[1])),
        ]
    smoothed_target.relative_camera_vector = _blend_vectors(last_target.relative_camera_vector, current_target.relative_camera_vector, alpha)
    smoothed_target.relative_world_vector = _blend_vectors(last_target.relative_world_vector, current_target.relative_world_vector, alpha)
    smoothed_target.world_target = _blend_vectors(last_target.world_target, current_target.world_target, alpha)
    if last_target.yaw_command is not None and current_target.yaw_command is not None:
        smoothed_target.yaw_command = float((1.0 - alpha) * last_target.yaw_command + alpha * current_target.yaw_command)
    smoothed_target.meta = dict(smoothed_target.meta)
    smoothed_target.meta['smoothed'] = True
    return smoothed_target


def _reuse_target_if_recent(
    response,
    *,
    last_target,
    last_target_ts: float | None,
    requested_target_label: str | None,
    hold_sec: float,
) -> tuple[object, bool, bool]:
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
    reused_target.meta['temporal_hold'] = True
    reused_target.meta['temporal_age_sec'] = round(age_sec, 3)
    response.target = reused_target
    return response, True, missed_detection


def _stabilize_target_switch(
    response,
    *,
    requested_target_label: str | None,
    last_target,
    candidate_target,
    candidate_count: int,
    switch_confirm: int,
):
    current_target = response.target
    if last_target is None or current_target is None:
        return response, current_target.model_copy(deep=True) if current_target is not None else None, 1 if current_target is not None else 0, False

    if requested_target_label is not None and last_target.label == requested_target_label and current_target.label != requested_target_label:
        reused_target = last_target.model_copy(deep=True)
        reused_target.meta = dict(reused_target.meta)
        reused_target.meta['switch_blocked'] = True
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
        reused_target.meta['switch_blocked'] = True
        reused_target.meta['switch_candidate'] = current_target.label
        reused_target.meta['switch_count'] = candidate_count
        response.target = reused_target
        return response, candidate_target, candidate_count, True

    return response, None, 0, False


def _hold_target_until_drop_threshold(
    response,
    *,
    last_target,
    requested_target_label: str | None,
    miss_count: int,
    drop_misses: int,
):
    current_target = response.target
    if last_target is None:
        return response, miss_count, False

    same_requested_label = requested_target_label is not None and last_target.label == requested_target_label
    target_missing = current_target is None
    target_mismatch = (
        current_target is not None
        and same_requested_label
        and current_target.label != requested_target_label
    )

    if target_missing or target_mismatch:
        miss_count += 1
    else:
        miss_count = 0

    if miss_count < max(drop_misses, 1) and (target_missing or target_mismatch):
        reused_target = last_target.model_copy(deep=True)
        reused_target.meta = dict(reused_target.meta)
        reused_target.meta['drop_hold'] = True
        reused_target.meta['drop_miss_count'] = miss_count
        response.target = reused_target
        return response, miss_count, True

    return response, miss_count, False


def _stabilize_target_motion(
    response,
    *,
    last_target,
    jump_candidate,
    jump_count: int,
    max_jump_px: float,
    switch_confirm: int,
    smooth_alpha: float,
):
    current_target = response.target
    if last_target is None or current_target is None:
        return response, jump_candidate, jump_count, False, False

    if current_target.label != last_target.label or current_target.type != last_target.type:
        return response, None, 0, False, False

    distance = _pixel_distance(last_target.pixel_center, current_target.pixel_center)
    if distance is None:
        return response, None, 0, False, False

    if distance <= max(max_jump_px, 1.0):
        response.target = _smooth_target_geometry(current_target, last_target, smooth_alpha)
        return response, None, 0, False, True

    if (
        jump_candidate is not None
        and jump_candidate.label == current_target.label
        and jump_candidate.type == current_target.type
        and _pixel_distance(jump_candidate.pixel_center, current_target.pixel_center) is not None
        and _pixel_distance(jump_candidate.pixel_center, current_target.pixel_center) <= max(max_jump_px, 1.0)
    ):
        jump_count += 1
    else:
        jump_candidate = current_target.model_copy(deep=True)
        jump_count = 1

    if jump_count < max(switch_confirm, 1):
        reused_target = last_target.model_copy(deep=True)
        reused_target.meta = dict(reused_target.meta)
        reused_target.meta['motion_blocked'] = True
        reused_target.meta['jump_px'] = round(distance, 2)
        reused_target.meta['jump_confirm'] = jump_count
        response.target = reused_target
        return response, jump_candidate, jump_count, True, False

    response.target = _smooth_target_geometry(current_target, last_target, smooth_alpha)
    return response, None, 0, False, True


def _decode_rgb(image_jpeg: bytes) -> np.ndarray:
    np_img = np.frombuffer(image_jpeg, np.uint8)
    frame_bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError('Failed to decode RGB image from ZeroMQ payload')
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _decode_depth(depth_png: bytes | None) -> np.ndarray | None:
    if not depth_png:
        return None
    np_img = np.frombuffer(depth_png, np.uint8)
    depth = cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError('Failed to decode depth image from ZeroMQ payload')
    return depth.astype(np.float32)


def _recv_loop(sock: zmq.Socket, state: SharedState) -> None:
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
            frame_rgb = _decode_rgb(payload['image_jpeg'])
            depth_map = _decode_depth(payload.get('depth_png'))
            if depth_map is not None:
                depth_scale = float(payload.get('depth_scale', 0.001))
                depth_map = depth_map * depth_scale
            intrinsics = CameraIntrinsics.model_validate(payload['intrinsics']) if payload.get('intrinsics') else None
            packet = LatestPacket(
                frame_rgb=frame_rgb,
                depth_map=depth_map,
                intrinsics=intrinsics,
                preferred_label=payload.get('preferred_label'),
                prefer_wall=bool(payload.get('prefer_wall')),
                task_prompt=payload.get('task_prompt'),
                sent_ts=payload.get('sent_ts'),
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


def _infer_loop(state: SharedState, args: argparse.Namespace, pose: DronePose | None) -> None:
    last_processed_seq = -1
    last_qwen_vision = None
    last_target = None
    last_target_ts: float | None = None
    candidate_target = None
    candidate_count = 0
    miss_count = 0
    jump_candidate = None
    jump_count = 0

    while not state.stop:
        with state.lock:
            packet = state.latest_packet

        if packet is None or packet.seq == last_processed_seq:
            time.sleep(0.01)
            continue

        started_at = time.time()
        try:
            run_qwen_now = (not args.disable_qwen) and (
                last_qwen_vision is None or packet.seq % max(args.qwen_every, 1) == 0
            )
            response = run_inference_pipeline(
                packet.frame_rgb,
                depth_map=packet.depth_map,
                camera_intrinsics=packet.intrinsics,
                pose=pose,
                preferred_label=args.preferred_label or packet.preferred_label,
                prefer_wall=bool(args.prefer_wall or packet.prefer_wall),
                task_prompt=args.task_prompt or packet.task_prompt,
                enable_dino=not args.disable_dino,
                enable_qwen=run_qwen_now,
                emit_runtime_artifacts=False,
            )
            if run_qwen_now:
                last_qwen_vision = {
                    'scene_description': response.vision.scene_description,
                    'projection_wall': response.vision.projection_wall.model_copy(deep=True),
                    'projection_surface': response.vision.projection_surface.model_copy(deep=True),
                }
            elif last_qwen_vision is not None:
                response.vision.scene_description = last_qwen_vision['scene_description']
                response.vision.projection_wall = last_qwen_vision['projection_wall'].model_copy(deep=True)
                response.vision.projection_surface = last_qwen_vision['projection_surface'].model_copy(deep=True)

            requested_target_label = _requested_target_label(packet, args)
            response, reused_previous_target, _ = _reuse_target_if_recent(
                response,
                last_target=last_target,
                last_target_ts=last_target_ts,
                requested_target_label=requested_target_label,
                hold_sec=max(args.target_hold_sec, 0.0),
            )
            response, candidate_target, candidate_count, blocked_switch = _stabilize_target_switch(
                response,
                requested_target_label=requested_target_label,
                last_target=last_target,
                candidate_target=candidate_target,
                candidate_count=candidate_count,
                switch_confirm=args.target_switch_confirm,
            )
            response, miss_count, held_for_drop = _hold_target_until_drop_threshold(
                response,
                last_target=last_target,
                requested_target_label=requested_target_label,
                miss_count=miss_count,
                drop_misses=args.target_drop_misses,
            )
            response, jump_candidate, jump_count, blocked_motion, smoothed_target = _stabilize_target_motion(
                response,
                last_target=last_target,
                jump_candidate=jump_candidate,
                jump_count=jump_count,
                max_jump_px=args.target_max_jump_px,
                switch_confirm=args.target_switch_confirm,
                smooth_alpha=args.target_smooth_alpha,
            )
            if reused_previous_target:
                response.vision.scene_description = response.vision.scene_description or 'Temporal hold on previous target.'
            elif blocked_switch:
                response.vision.scene_description = response.vision.scene_description or 'Target switch confirmation in progress.'
            elif held_for_drop:
                response.vision.scene_description = response.vision.scene_description or 'Keeping target through temporary detection drop.'
            elif blocked_motion:
                response.vision.scene_description = response.vision.scene_description or 'Keeping target through large position jump.'
            elif smoothed_target:
                response.vision.scene_description = response.vision.scene_description or 'Smoothing target motion.'

            if response.target is not None and not response.target.meta.get('temporal_hold') and not response.target.meta.get('switch_blocked') and not response.target.meta.get('drop_hold'):
                last_target = response.target.model_copy(deep=True)
                last_target_ts = time.time()
                miss_count = 0

            infer_latency_ms = (time.time() - started_at) * 1000.0
            overlay_payload = response.model_dump(exclude={'vision': {'raw_text'}})
            if args.task_prompt or packet.task_prompt:
                overlay_payload['task_prompt'] = args.task_prompt or packet.task_prompt
            target = response.target.model_dump() if response.target is not None else None
            logger.info('zmq_infer latency_ms=%.1f target=%s', infer_latency_ms, json.dumps(target, ensure_ascii=False))
            logger.info('inference_result=%s', json.dumps(overlay_payload, ensure_ascii=False))
            with state.lock:
                state.latest_overlay_payload = overlay_payload
                state.latest_infer_latency_ms = infer_latency_ms
                state.processed_count += 1
                state.error_text = None
                last_processed_seq = packet.seq
        except Exception as exc:
            with state.lock:
                state.error_text = str(exc)
            logger.exception('ZeroMQ inference failed: %s', exc)
            last_processed_seq = packet.seq


def _display_loop(state: SharedState) -> None:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    while not state.stop:
        with state.lock:
            packet = state.latest_packet
            overlay_payload = state.latest_overlay_payload
            infer_latency_ms = state.latest_infer_latency_ms
            transport_latency_ms = state.latest_transport_latency_ms
            received_count = state.received_count
            processed_count = state.processed_count
            error_text = state.error_text

        if packet is None:
            time.sleep(0.01)
            continue

        frame_bgr = cv2.cvtColor(packet.frame_rgb, cv2.COLOR_RGB2BGR)
        rendered = draw_vision_result(frame_bgr, overlay_payload or {})
        if DISPLAY_SCALE != 1.0:
            rendered = cv2.resize(rendered, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_CUBIC)

        status = f'recv={received_count} infer={processed_count}'
        if transport_latency_ms is not None:
            status += f' | net={transport_latency_ms:.1f}ms'
        if infer_latency_ms is not None:
            status += f' | infer={infer_latency_ms:.1f}ms'
        cv2.putText(rendered, status, (12, rendered.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        if error_text:
            cv2.putText(rendered, error_text[:110], (12, rendered.shape[0] - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, rendered)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            state.stop = True
            break

    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.RCVHWM, 1)
    sock.bind(args.bind)

    pose = DronePose.model_validate_json(args.drone_pose) if args.drone_pose else None
    state = SharedState(lock=threading.Lock())

    logger.info('ZeroMQ receiver listening on %s', args.bind)

    recv_thread = threading.Thread(target=_recv_loop, args=(sock, state), daemon=True)
    infer_thread = threading.Thread(target=_infer_loop, args=(state, args, pose), daemon=True)
    recv_thread.start()
    infer_thread.start()

    try:
        if args.no_display:
            while not state.stop:
                time.sleep(0.2)
        else:
            _display_loop(state)
    except KeyboardInterrupt:
        pass
    finally:
        state.stop = True
        recv_thread.join(timeout=1.0)
        infer_thread.join(timeout=1.0)
        sock.close(0)


if __name__ == '__main__':
    main()
