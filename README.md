# HoverAI Qwen2.5-VL

Indoor drone demo: Orange Pi streams RGB + depth from RealSense D435i → server runs Grounding DINO + Qwen2.5-VL → returns detected objects, 3D target, and free wall for projection.

## Quick Start

### 1. Activate environment (server)

```bash
cd /media/imit-learn/ISR_2T3/OMNI_AI_VLM
source scripts/activate_vlm_hoverai.sh
```

### 2. Start server (ZeroMQ)

DINO + Qwen:
```bash
python -m server.zmq_receiver --bind tcp://*:5555 --qwen-every 5
```

DINO only (faster, lower latency):
```bash
python -m server.zmq_receiver --bind tcp://*:5555 --disable-qwen
```

### 3. Start client (Orange Pi)

```bash
cd ~/HoverAI_VLM/OMNI_AI_VLM && source ../.venv/bin/activate && export PYTHONPATH=$PWD && python tools/d435i_zmq_sender.py --server-ip 192.168.50.185 --port 5555 --infer-every 1 --infer-width 448 --jpeg-quality 60 --send-depth
```

With task prompt (wall projection):
```bash
python tools/d435i_zmq_sender.py --server-ip 192.168.50.185 --port 5555 --infer-every 1 --infer-width 448 --jpeg-quality 60 --send-depth --task-prompt "find headphones and project on wall"
```

---

## Switching target objects at runtime

Press keys in the **server preview window** while running:

| Key | Object     | Key | Object    |
|-----|------------|-----|-----------|
| `p` | person     | `r` | robot     |
| `c` | cube       | `t` | table     |
| `b` | ball       | `l` | laptop    |
| `h` | headphones | `d` | door      |
| `f` | fish       | `n` | plant     |
| `e` | bottle     | `k` | book      |
| `u` | cup        | `0` | all (reset) |
| `q` | quit       |     |           |

Current target is shown at the bottom of the preview window.

---

## Data collection (detection range & stability test)

Start server with `--record`:
```bash
python -m server.zmq_receiver --bind tcp://*:5555 --disable-qwen --record detections.jsonl
```

**Test procedure:**
1. Place object at 0.5 m → press target key → wait 10 s
2. Move to 1.0 m → wait 10 s
3. Repeat at 1.5 m, 2.0 m, 3.0 m
4. Press `0` to switch to next object, repeat
5. Stop with `Ctrl+C`

---

## Analytics

```bash
# All objects
python tools/analyze_detections.py detections.jsonl

# Single object
python tools/analyze_detections.py detections.jsonl --label cube

# Limit depth range
python tools/analyze_detections.py detections.jsonl --label cube --max-depth 3.0
```

Report shows: detection rate, max detection distance, depth stability (CV), confidence stats, detection run lengths.

---

## HTTP path (alternative to ZeroMQ)

Server:
```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Client (Orange Pi):
```bash
python tools/d435i_preview.py --server-url http://192.168.50.185:8000/infer --infer-every 15 --send-depth
```
