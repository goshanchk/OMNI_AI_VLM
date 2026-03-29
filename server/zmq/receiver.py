from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time

import cv2
import zmq

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.schemas import DronePose
from common.visualization import draw_vision_result
from server.main import run_inference_pipeline
from server.stats_recorder import append_detection_record, build_detection_record
from server.task_parser import parse_task_prompt
from server.zmq.runtime import (
    LABEL_KEYS,
    apply_keyboard_label,
    current_prefer_wall,
    effective_projection,
    runtime_status_line,
    stdin_command_loop,
)
from server.zmq.state import SharedState
from server.zmq.tracking import (
    TemporalTrackingState,
    hold_target_until_drop_threshold,
    needs_tracking_reset,
    reset_tracking_state,
    reuse_target_if_recent,
    stabilize_target_motion,
    stabilize_target_switch,
)
from server.zmq.transport import recv_loop

logger = logging.getLogger("hoverai.zmq_receiver")
WINDOW_NAME = "HoverAI ZeroMQ Live Preview"
DISPLAY_SCALE = float(os.getenv("HOVERAI_PREVIEW_SCALE", "1.35"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HoverAI ZeroMQ receiver for low-latency inference")
    parser.add_argument("--bind", default="tcp://*:5555")
    parser.add_argument("--preferred-label", default=None)
    parser.add_argument("--prefer-wall", action="store_true")
    parser.add_argument("--task-prompt", default=None)
    parser.add_argument("--disable-dino", action="store_true")
    parser.add_argument("--disable-qwen", action="store_true")
    parser.add_argument("--qwen-every", type=int, default=3, help="Run Qwen every N processed frames and reuse last projection in between")
    parser.add_argument("--target-hold-sec", type=float, default=1.2, help="Keep the last good target briefly when detection drops out")
    parser.add_argument("--target-switch-confirm", type=int, default=2, help="Require N consecutive frames before switching to a different target")
    parser.add_argument("--target-drop-misses", type=int, default=3, help="Drop the current target only after N consecutive missed frames")
    parser.add_argument("--target-max-jump-px", type=float, default=34.0, help="Allow only small target motion between frames before requiring confirmation")
    parser.add_argument("--target-smooth-alpha", type=float, default=0.35, help="Blend factor for target smoothing, smaller is steadier")
    parser.add_argument("--drone-pose", default=None, help='JSON string like {"x":0,"y":0,"z":1.2,"yaw":0.0}')
    parser.add_argument("--no-display", action="store_true", help="Disable live server preview window")
    parser.add_argument("--record", default=None, metavar="PATH", help="Append detection records to a JSONL file for later analysis")
    return parser.parse_args()


def _current_runtime_inputs(state: SharedState, args: argparse.Namespace) -> tuple[str | None, str | None, str | None, bool]:
    with state.lock:
        runtime_label = state.runtime_label
        runtime_prompt = state.runtime_prompt
    preferred_label = runtime_label if runtime_label is not None else args.preferred_label
    task_prompt = runtime_prompt if runtime_prompt is not None else args.task_prompt
    prefer_wall = current_prefer_wall(state, args)
    projection = effective_projection(state, args)
    return preferred_label, task_prompt, projection, prefer_wall


def _apply_cached_qwen_projection(response, cached_qwen_vision: dict | None) -> None:
    if cached_qwen_vision is None:
        return
    response.vision.scene_description = cached_qwen_vision["scene_description"]
    response.vision.projection_wall = cached_qwen_vision["projection_wall"].model_copy(deep=True)
    response.vision.projection_surface = cached_qwen_vision["projection_surface"].model_copy(deep=True)


def _update_scene_description(
    response,
    *,
    reused_previous_target: bool,
    blocked_switch: bool,
    held_for_drop: bool,
    blocked_motion: bool,
    smoothed_target: bool,
) -> None:
    if reused_previous_target:
        response.vision.scene_description = response.vision.scene_description or "Temporal hold on previous target."
    elif blocked_switch:
        response.vision.scene_description = response.vision.scene_description or "Target switch confirmation in progress."
    elif held_for_drop:
        response.vision.scene_description = response.vision.scene_description or "Keeping target through temporary detection drop."
    elif blocked_motion:
        response.vision.scene_description = response.vision.scene_description or "Keeping target through large position jump."
    elif smoothed_target:
        response.vision.scene_description = response.vision.scene_description or "Smoothing target motion."


def _build_overlay_payload(response, task_prompt: str | None, projection: str | None) -> dict:
    overlay_payload = response.model_dump(exclude={"vision": {"raw_text"}})
    if task_prompt:
        overlay_payload["task_prompt"] = task_prompt
    overlay_payload["runtime_projection"] = projection
    return overlay_payload


def _persist_record(
    args: argparse.Namespace,
    *,
    packet_seq: int,
    preferred_label: str | None,
    infer_latency_ms: float,
    response,
) -> None:
    if not args.record:
        return
    record = build_detection_record(
        ts=time.time(),
        seq=packet_seq,
        runtime_label=preferred_label,
        infer_ms=infer_latency_ms,
        response=response,
    )
    try:
        append_detection_record(args.record, record)
    except OSError as exc:
        logger.warning("record write failed: %s", exc)


def _infer_loop(state: SharedState, args: argparse.Namespace, pose: DronePose | None) -> None:
    last_processed_seq = -1
    tracking = TemporalTrackingState()

    while not state.stop:
        with state.lock:
            packet = state.latest_packet

        if packet is None or packet.seq == last_processed_seq:
            time.sleep(0.01)
            continue

        preferred_label, task_prompt, projection, prefer_wall = _current_runtime_inputs(state, args)
        if needs_tracking_reset(tracking, preferred_label, task_prompt, prefer_wall):
            reset_tracking_state(tracking, preferred_label, task_prompt, prefer_wall)

        started_at = time.time()
        try:
            run_qwen_now = (not args.disable_qwen) and (
                tracking.last_qwen_vision is None or packet.seq % max(args.qwen_every, 1) == 0
            )
            response = run_inference_pipeline(
                packet.frame_rgb,
                depth_map=packet.depth_map,
                camera_intrinsics=packet.intrinsics,
                pose=pose,
                preferred_label=preferred_label,
                prefer_wall=prefer_wall,
                task_prompt=task_prompt,
                enable_dino=not args.disable_dino,
                enable_qwen=run_qwen_now,
                emit_runtime_artifacts=False,
            )
            if run_qwen_now:
                tracking.last_qwen_vision = {
                    "scene_description": response.vision.scene_description,
                    "projection_wall": response.vision.projection_wall.model_copy(deep=True),
                    "projection_surface": response.vision.projection_surface.model_copy(deep=True),
                }
            else:
                _apply_cached_qwen_projection(response, tracking.last_qwen_vision)

            requested_target_label = preferred_label if preferred_label else parse_task_prompt(task_prompt).target_label
            response, reused_previous_target, _ = reuse_target_if_recent(
                response,
                last_target=tracking.last_target,
                last_target_ts=tracking.last_target_ts,
                requested_target_label=requested_target_label,
                hold_sec=max(args.target_hold_sec, 0.0),
            )
            response, tracking.candidate_target, tracking.candidate_count, blocked_switch = stabilize_target_switch(
                response,
                requested_target_label=requested_target_label,
                last_target=tracking.last_target,
                candidate_target=tracking.candidate_target,
                candidate_count=tracking.candidate_count,
                switch_confirm=args.target_switch_confirm,
            )
            response, tracking.miss_count, held_for_drop = hold_target_until_drop_threshold(
                response,
                last_target=tracking.last_target,
                requested_target_label=requested_target_label,
                miss_count=tracking.miss_count,
                drop_misses=args.target_drop_misses,
            )
            response, tracking.jump_candidate, tracking.jump_count, blocked_motion, smoothed_target = stabilize_target_motion(
                response,
                last_target=tracking.last_target,
                jump_candidate=tracking.jump_candidate,
                jump_count=tracking.jump_count,
                max_jump_px=args.target_max_jump_px,
                switch_confirm=args.target_switch_confirm,
                smooth_alpha=args.target_smooth_alpha,
            )
            _update_scene_description(
                response,
                reused_previous_target=reused_previous_target,
                blocked_switch=blocked_switch,
                held_for_drop=held_for_drop,
                blocked_motion=blocked_motion,
                smoothed_target=smoothed_target,
            )

            if (
                response.target is not None
                and not response.target.meta.get("temporal_hold")
                and not response.target.meta.get("switch_blocked")
                and not response.target.meta.get("drop_hold")
            ):
                tracking.last_target = response.target.model_copy(deep=True)
                tracking.last_target_ts = time.time()
                tracking.miss_count = 0

            infer_latency_ms = (time.time() - started_at) * 1000.0
            _persist_record(
                args,
                packet_seq=packet.seq,
                preferred_label=preferred_label,
                infer_latency_ms=infer_latency_ms,
                response=response,
            )
            overlay_payload = _build_overlay_payload(response, task_prompt, projection)
            target = response.target.model_dump() if response.target is not None else None
            logger.info("zmq_infer latency_ms=%.1f target=%s", infer_latency_ms, json.dumps(target, ensure_ascii=False))
            logger.info("inference_result=%s", json.dumps(overlay_payload, ensure_ascii=False))
            with state.lock:
                state.latest_overlay_payload = overlay_payload
                state.latest_infer_latency_ms = infer_latency_ms
                state.processed_count += 1
                state.error_text = None
            last_processed_seq = packet.seq
        except Exception as exc:
            with state.lock:
                state.error_text = str(exc)
            logger.exception("ZeroMQ inference failed: %s", exc)
            last_processed_seq = packet.seq


def _display_loop(state: SharedState, args: argparse.Namespace) -> None:
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
            command_hint = state.command_hint

        if packet is None:
            time.sleep(0.01)
            continue

        effective_label, effective_prompt, effective_projection_target, _ = _current_runtime_inputs(state, args)
        frame_bgr = cv2.cvtColor(packet.frame_rgb, cv2.COLOR_RGB2BGR)
        rendered = draw_vision_result(frame_bgr, overlay_payload or {})
        if DISPLAY_SCALE != 1.0:
            rendered = cv2.resize(rendered, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_CUBIC)

        status = f"recv={received_count} infer={processed_count}"
        if transport_latency_ms is not None:
            status += f" | net={transport_latency_ms:.1f}ms"
        if infer_latency_ms is not None:
            status += f" | infer={infer_latency_ms:.1f}ms"
        cv2.putText(rendered, status, (12, rendered.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        label_text = f"target: {effective_label or 'all'}  [0=all p=person c=cube b=ball h=headphones r=robot f=fish t=table]"
        cv2.putText(rendered, label_text, (12, rendered.shape[0] - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 60), 1, cv2.LINE_AA)

        prompt_text = f"prompt: {(effective_prompt or '-')[:95]}"
        cv2.putText(rendered, prompt_text, (12, rendered.shape[0] - 68), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 220, 220), 1, cv2.LINE_AA)
        projection_text = f"projection: {effective_projection_target or '-'}"
        cv2.putText(rendered, projection_text, (12, rendered.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 220, 220), 1, cv2.LINE_AA)

        if command_hint:
            cv2.putText(rendered, command_hint[:110], (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 240, 255), 1, cv2.LINE_AA)
        if error_text:
            cv2.putText(rendered, error_text[:110], (12, rendered.shape[0] - 86), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, rendered)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            state.stop = True
            break
        if key in LABEL_KEYS:
            with state.lock:
                apply_keyboard_label(state, LABEL_KEYS[key])
            logger.info("runtime_label switched to %s", LABEL_KEYS[key])

    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.RCVHWM, 1)
    sock.bind(args.bind)

    pose = DronePose.model_validate_json(args.drone_pose) if args.drone_pose else None
    state = SharedState(lock=threading.Lock())

    logger.info("ZeroMQ receiver listening on %s", args.bind)
    logger.info("runtime status: %s", runtime_status_line(state, args))

    recv_thread = threading.Thread(target=recv_loop, args=(sock, state), daemon=True)
    infer_thread = threading.Thread(target=_infer_loop, args=(state, args, pose), daemon=True)
    stdin_thread = threading.Thread(target=stdin_command_loop, args=(state, args), daemon=True)
    recv_thread.start()
    infer_thread.start()
    stdin_thread.start()

    try:
        if args.no_display:
            while not state.stop:
                time.sleep(0.2)
        else:
            _display_loop(state, args)
    except KeyboardInterrupt:
        pass
    finally:
        state.stop = True
        recv_thread.join(timeout=1.0)
        infer_thread.join(timeout=1.0)
        sock.close(0)


if __name__ == "__main__":
    main()
