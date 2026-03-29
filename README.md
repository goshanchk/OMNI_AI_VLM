# HoverAI Qwen2.5-VL

Orange Pi sends RGB + depth over ZeroMQ, and the server returns detected objects, target, and projection surface.

## Server

Start the server:
```bash
cd /media/imit-learn/ISR_2T3/OMNI_AI_VLM && source scripts/activate_vlm_hoverai.sh && python -m server.zmq.receiver --bind tcp://*:5555 --qwen-every 5
```
## Client

Start the client on Orange Pi from the same project root:
```bash
cd ~/HoverAI_VLM/OMNI_AI_VLM && source scripts/activate_vlm_hoverai.sh && python tools/d435i_zmq_sender.py --server-ip 192.168.50.185 --port 5555 --infer-every 1 --infer-width 448 --jpeg-quality 60 --send-depth
```

## Runtime Control

Change the target object during video using keys in the server preview window:

- `p` person
- `c` cube
- `b` ball
- `h` headphones
- `f` fish
- `r` robot
- `t` table
- `l` laptop
- `d` door
- `n` plant
- `k` book
- `u` cup
- `e` bottle
- `0` reset to all objects
- `q` quit

Change the prompt from the server terminal:
```text
status
object person
projection wall
projection screen
prompt find person and project on wall
clear object
clear projection
clear prompt
reset
```

`projection wall` selects wall projection, `projection screen` selects the drone screen.

## Data collection (detection range & stability)

Start server with `--record`:
```bash
cd /media/imit-learn/ISR_2T3/OMNI_AI_VLM && source scripts/activate_vlm_hoverai.sh && python -m server.zmq.receiver --bind tcp://*:5555 --disable-qwen --record detections.jsonl
```

## Analytics

```bash
# All objects
python tools/analyze_detections.py detections.jsonl

# Single object
python tools/analyze_detections.py detections.jsonl --label headphones

# Limit depth range
python tools/analyze_detections.py detections.jsonl --label cube --max-depth 3.0
```

Report shows: detection rate, max detection distance, depth stability (CV), confidence, detection run lengths.
