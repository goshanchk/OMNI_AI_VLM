"""
Analyze detection records written by zmq_receiver.py --record.

Usage:
    python tools/analyze_detections.py detections.jsonl
    python tools/analyze_detections.py detections.jsonl --label cube
    python tools/analyze_detections.py detections.jsonl --min-depth 0.3 --max-depth 5.0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def load_records(path: str, label_filter: str | None, min_depth: float, max_depth: float) -> list[dict]:
    records = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if label_filter and r.get('runtime_label') != label_filter and r.get('label') != label_filter:
                continue
            d = r.get('depth_m')
            if d is not None and (d < min_depth or d > max_depth):
                r['depth_m'] = None
            records.append(r)
    return records


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float('nan')


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return float('nan')
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _percentile(vals: list[float], p: float) -> float:
    if not vals:
        return float('nan')
    s = sorted(vals)
    idx = (len(s) - 1) * p / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def _consecutive_runs(detected: list[bool]) -> list[int]:
    runs, cur = [], 0
    for d in detected:
        if d:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def _depth_histogram(depths: list[float], bins: int = 10) -> str:
    if not depths:
        return '  (no data)'
    lo, hi = min(depths), max(depths)
    if hi == lo:
        return f'  all at {lo:.2f} m'
    width = (hi - lo) / bins
    counts = [0] * bins
    for d in depths:
        idx = min(int((d - lo) / width), bins - 1)
        counts[idx] += 1
    max_count = max(counts)
    bar_width = 20
    lines = []
    for i, c in enumerate(counts):
        lo_b = lo + i * width
        hi_b = lo_b + width
        bar = '█' * int(bar_width * c / max_count) if max_count else ''
        lines.append(f'  {lo_b:5.2f}-{hi_b:5.2f}m │{bar:<{bar_width}}│ {c}')
    return '\n'.join(lines)


def analyze(records: list[dict]) -> None:
    if not records:
        print('No records found.')
        return

    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        label = r.get('runtime_label') or r.get('label') or 'unknown'
        by_label[label].append(r)

    total_frames = len(records)
    duration_s = records[-1]['ts'] - records[0]['ts'] if len(records) > 1 else 0.0
    fps = total_frames / duration_s if duration_s > 0 else 0.0

    print('=' * 62)
    print(f'  Detection Analysis Report')
    print(f'  Total frames : {total_frames}')
    print(f'  Duration     : {duration_s:.1f} s  (~{fps:.1f} fps)')
    print('=' * 62)

    for label, recs in sorted(by_label.items()):
        detected_flags = [
            r.get('label') is not None and not r.get('is_hold', False)
            for r in recs
        ]
        real_depths = [
            r['depth_m']
            for r in recs
            if r.get('depth_m') is not None and not r.get('is_hold', False)
        ]
        hold_depths = [
            r['depth_m']
            for r in recs
            if r.get('depth_m') is not None and r.get('is_hold', False)
        ]
        confidences = [
            r['confidence']
            for r in recs
            if r.get('confidence') is not None and not r.get('is_hold', False)
        ]

        n_total = len(recs)
        n_detected = sum(detected_flags)
        det_rate = 100.0 * n_detected / n_total if n_total else 0.0

        runs = _consecutive_runs(detected_flags)
        mean_run = _mean([float(r) for r in runs]) if runs else 0.0
        max_run = max(runs) if runs else 0
        stability = 100.0 * mean_run / max_run if max_run else 0.0

        print(f'\n── Label: {label}  ({n_total} frames) ──')
        print(f'  Detection rate   : {n_detected}/{n_total} = {det_rate:.1f}%')
        print(f'  Holds (temporal) : {len(hold_depths)} frames')

        if real_depths:
            print(f'\n  Depth (real detections, no hold):')
            print(f'    min   : {min(real_depths):.2f} m')
            print(f'    max   : {max(real_depths):.2f} m   ← max detection distance')
            print(f'    mean  : {_mean(real_depths):.2f} m')
            print(f'    std   : {_std(real_depths):.2f} m')
            print(f'    p25   : {_percentile(real_depths, 25):.2f} m')
            print(f'    p75   : {_percentile(real_depths, 75):.2f} m')
            cv = _std(real_depths) / _mean(real_depths) if _mean(real_depths) else float('nan')
            print(f'    CV    : {cv:.3f}  (std/mean, lower = steadier depth)')
            print(f'\n  Depth histogram:')
            print(_depth_histogram(real_depths))
        else:
            print('  Depth: no real detections with depth data.')

        if confidences:
            print(f'\n  Confidence:')
            print(f'    min  : {min(confidences):.3f}')
            print(f'    max  : {max(confidences):.3f}')
            print(f'    mean : {_mean(confidences):.3f}')

        print(f'\n  Temporal stability:')
        print(f'    consecutive detection runs : {len(runs)}')
        print(f'    longest run                : {max_run} frames')
        print(f'    mean run length            : {mean_run:.1f} frames')
        print(f'    stability score            : {stability:.0f}%  (mean/max run)')

    print('\n' + '=' * 62)


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze HoverAI detection JSONL records')
    parser.add_argument('file', help='Path to the JSONL file written by zmq_receiver --record')
    parser.add_argument('--label', default=None, help='Filter by runtime_label or detected label')
    parser.add_argument('--min-depth', type=float, default=0.1, help='Ignore depth values below this (meters)')
    parser.add_argument('--max-depth', type=float, default=10.0, help='Ignore depth values above this (meters)')
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f'File not found: {path}', file=sys.stderr)
        sys.exit(1)

    records = load_records(str(path), args.label, args.min_depth, args.max_depth)
    analyze(records)


if __name__ == '__main__':
    main()
