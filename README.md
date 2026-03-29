# HoverAI Qwen2.5-VL MVP

Minimal MVP for an indoor drone demo:

- Orange Pi captures RGB frames and optionally depth.
- The server runs `Qwen/Qwen2.5-VL-3B-Instruct` and `Grounding DINO`.
- The server returns structured scene understanding, object detections, and a projection target.
- If depth and intrinsics are provided, the server converts the 2D target into a 3D camera/world target.
- If depth is not provided, the server falls back to a `yaw_command`.
- Orange Pi forwards the selected target to the flight stack bridge.

## Repository Layout

- `server/main.py` — FastAPI API entrypoint.
- `server/qwen_inference.py` — Qwen model loading and inference.
- `server/dino_detector.py` — Grounding DINO inference.
- `server/task_parser.py` — task prompt parsing.
- `common/geometry.py` — 2D to 3D projection and target selection.
- `common/schemas.py` — Pydantic schemas shared by server and clients.
- `common/visualization.py` — overlay rendering for preview.
- `orange_pi/client.py` — simple RGB client using `cv2.VideoCapture`.
- `orange_pi/flight_bridge.py` — stub bridge to a flight stack.
- `orange_pi/vicon_bridge.py` — stub Vicon adapter.
- `tools/d435i_preview.py` — RealSense D435i live client with RGB, depth, intrinsics, and preview window.

## Server Setup

### Option 1: Python venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option 2: Conda

```bash
source scripts/activate_vlm_hoverai.sh
```

### Environment check

```bash
python -c "import torch, transformers, fastapi, regex; print(torch.__version__, transformers.__version__)"
```

## Run the Server

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Expected response shape:

```json
{
  "status": "ok",
  "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
  "dino_model_id": "IDEA-Research/grounding-dino-base"
}
```

## Model Requirements

The server needs access to these Hugging Face models:

- `Qwen/Qwen2.5-VL-3B-Instruct`
- `IDEA-Research/grounding-dino-base`

If they are not cached locally, the first run will download them from Hugging Face.

## Orange Pi Client Modes

There are two client paths in this repo:

### 1. Simple RGB client

Use `orange_pi/client.py` for a basic USB camera or a demo flow with an optional static depth file:

```bash
python -m orange_pi.client \
  --server-url http://SERVER_IP:8000/infer \
  --camera-index 0 \
  --preferred-label person \
  --drone-pose '{"x":0.0,"y":0.0,"z":1.2,"yaw":0.0}'
```

With intrinsics:

```bash
python -m orange_pi.client \
  --server-url http://SERVER_IP:8000/infer \
  --camera-index 0 \
  --intrinsics '{"fx":525.0,"fy":525.0,"cx0":319.5,"cy0":239.5}'
```

With a static depth file:

```bash
python -m orange_pi.client \
  --server-url http://SERVER_IP:8000/infer \
  --camera-index 0 \
  --depth-image /path/to/depth.png \
  --intrinsics '{"fx":525.0,"fy":525.0,"cx0":319.5,"cy0":239.5}'
```

### 2. Intel RealSense D435i client

Use `tools/d435i_preview.py` when Orange Pi is connected to a RealSense D435i and must stream live RGB + depth to the server.

This script:

- reads RGB and depth from the camera,
- extracts RealSense intrinsics automatically,
- sends frames to the server every `N` frames,
- renders detections, target, and projection overlays in a local OpenCV window.

Basic run:

```bash
python tools/d435i_preview.py \
  --server-url http://SERVER_IP:8000/infer \
  --infer-every 15 \
  --send-depth
```

Smoother preview mode:

```bash
python tools/d435i_preview.py \
  --server-url http://SERVER_IP:8000/infer \
  --infer-every 45 \
  --infer-width 448 \
  --jpeg-quality 80 \
  --send-depth
```

Task-driven example:

```bash
python tools/d435i_preview.py \
  --server-url http://SERVER_IP:8000/infer \
  --infer-every 15 \
  --send-depth \
  --task-prompt "find a person and project on wall"
```

## Minimal Orange Pi Setup Without Conda

If Orange Pi is used only as a thin client and all heavy inference runs on the server:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip libglib2.0-0 libgl1 libusb-1.0-0
cd ~/HoverAI_VLM
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pillow requests opencv-python pyrealsense2
```

Then run the D435i client from the project root:

```bash
cd ~/HoverAI_VLM
source .venv/bin/activate
export PYTHONPATH=$PWD
python tools/d435i_preview.py \
  --server-url http://SERVER_IP:8000/infer \
  --infer-every 15 \
  --send-depth
```

Important:

- Orange Pi must have the full repo layout, including `common/` and `tools/`.
- `127.0.0.1` works only when client and server run on the same machine.
- If Orange Pi and the server are different machines, use the server IP in `--server-url`.

## Example Server Response

```json
{
  "vision": {
    "scene_description": "office with table and wall",
    "objects": [
      {
        "label": "chair",
        "bbox_2d": [100, 120, 220, 340],
        "center_2d": [160, 230],
        "confidence": 0.91,
        "source": "dino"
      }
    ],
    "projection_wall": {
      "found": true,
      "bbox_2d": [300, 40, 620, 400],
      "center_2d": [460, 220]
    },
    "projection_surface": {
      "found": true,
      "surface_type": "wall",
      "bbox_2d": [300, 40, 620, 400],
      "center_2d": [460, 220],
      "is_free": true,
      "suitability": 0.84,
      "reason": "Large clear wall patch."
    }
  },
  "target": {
    "type": "object",
    "label": "chair",
    "pixel_center": [160, 230],
    "relative_camera_vector": [0.1, -0.05, 2.3],
    "relative_world_vector": [0.1, -0.05, 2.3],
    "world_target": [1.4, 0.2, 3.5],
    "yaw_command": null,
    "source": "depth",
    "meta": {
      "image_shape": [480, 640]
    }
  }
}
```

## Integration Notes

- Upload depth as a second multipart file named `depth`.
- Send `intrinsics` and `drone_pose` as JSON strings in form-data.
- Replace `print(...)` in `orange_pi/flight_bridge.py` with real MAVSDK, MAVROS, or PX4 integration.
- Replace `orange_pi/vicon_bridge.py` with your real Vicon pose source if needed.

## Quick Smoke Checks

Server import:

```bash
python -c "import server.main; print(server.main.app.title)"
```

Compile all Python files:

```bash
python -m compileall common orange_pi server tools
```

## Known Constraints

- The server requires model weights to be available locally or downloadable from Hugging Face.
- `tools/d435i_preview.py` requires `pyrealsense2` and a connected RealSense D435i.
- `orange_pi/client.py` does not read live RealSense depth; use `tools/d435i_preview.py` for D435i.
