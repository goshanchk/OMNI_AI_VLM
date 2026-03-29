# HoverAI Qwen2.5-VL

Orange Pi sends RGB + depth over ZeroMQ, the server runs detection and returns objects, target, and projection surface.

## Server

Activate the environment:
```bash
cd /media/imit-learn/ISR_2T3/OMNI_AI_VLM
source scripts/activate_vlm_hoverai.sh
```

Start the server:
```bash
python -m server.zmq.receiver --bind tcp://*:5555 --qwen-every 5
```

Start the server with detection recording:
```bash
python -m server.zmq.receiver --bind tcp://*:5555 --disable-qwen --record detections.jsonl
```

## Client

Start the client on Orange Pi:
```bash
cd ~/HoverAI_VLM/OMNI_AI_VLM
source ../.venv/bin/activate
export PYTHONPATH=$PWD
python tools/d435i_zmq_sender.py --server-ip 192.168.50.185 --port 5555 --infer-every 1 --infer-width 448 --jpeg-quality 60 --send-depth
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
