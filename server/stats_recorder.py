from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("hoverai.stats_recorder")


def build_detection_record(
    *,
    ts: float,
    seq: int,
    runtime_label: str | None,
    infer_ms: float,
    response: Any,
) -> dict[str, Any]:
    target = response.target
    depth_m = None
    confidence = None
    bbox_2d = None
    is_hold = False

    if target is not None:
        rel_vec = target.relative_camera_vector
        depth_m = round(float(rel_vec[2]), 3) if rel_vec and len(rel_vec) == 3 and rel_vec[2] > 0 else None
        is_hold = bool(target.meta.get('temporal_hold') or target.meta.get('drop_hold'))
        for obj in response.vision.objects:
            if obj.label == target.label:
                confidence = round(float(obj.confidence), 4) if obj.confidence is not None else None
                bbox_2d = obj.bbox_2d
                break

    return {
        'ts': round(ts, 3),
        'seq': seq,
        'runtime_label': runtime_label,
        'label': target.label if target is not None else None,
        'depth_m': depth_m,
        'confidence': confidence,
        'bbox_2d': bbox_2d,
        'source': target.source if target is not None else None,
        'is_hold': is_hold,
        'image_shape': response.vision.image_shape,
        'infer_ms': round(infer_ms, 1),
    }


def append_detection_record(path: str, record: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record) + '\n')


def persist_record(
    args: argparse.Namespace,
    *,
    packet_seq: int,
    preferred_label: str | None,
    infer_latency_ms: float,
    response: Any,
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
