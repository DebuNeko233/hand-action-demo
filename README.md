# Hand Action Demo

Real-time local-camera demo: **OpenCV → MediaPipe 21-point hand skeleton → normalized 66-D temporal feature → PyTorch LSTM → action + confidence**.

## Actions
`idle`, `grab`, `release`, `move`, `rotate`, `wipe`.

> `grab` in V1 means an open→closed hand-motion pattern, **not proof that an object was physically grasped**. Object-aware semantics require an object detector / hand-object relation stage later.

## Environment
Python 3.10–3.12 recommended. MediaPipe is pinned to `1.0.1` in this demo.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python tools/download_models.py
```

The MediaPipe Tasks HandLandmarker requires a local `.task` model. `download_models.py` fetches the official model into `models/hand_landmarker.task`.

## Collect data

```bash
python tools/collect_data.py --action idle --samples 200
python tools/collect_data.py --action grab --samples 200
python tools/collect_data.py --action release --samples 200
python tools/collect_data.py --action move --samples 200
python tools/collect_data.py --action rotate --samples 200
python tools/collect_data.py --action wipe --samples 200
```

Keys: `SPACE` record/pause, `S` save current 30-frame sample, `R` reset buffer, `Q` quit. Add `--auto` for automatic samples; default stride is 30 frames.

Suggested collection diversity: change position, distance, speed, amplitude and direction. For `wipe`, perform 2–3 reciprocating sweeps; for `move`, keep hand shape relatively stable; for `rotate`, rotate palm/wrist clearly.

## Train

```bash
python tools/inspect_dataset.py
python tools/train.py
```

Best checkpoint: `models/hand_action_lstm.pth`. Training curve: `outputs/training_curve.png`.

## Run

```bash
python main.py --camera 0
python main.py --camera 0 --debug
```

The model waits until a 30-frame sequence is available. Predictions run every 3 frames and use confidence threshold + recent majority smoothing. Five consecutive no-hand frames clear temporal state.

## Feature definition
Each frame produces 66 float values:
- 63 = 21×XYZ landmarks centered at wrist and scaled by wrist→middle-MCP distance.
- 3 = raw normalized-image wrist displacement `(dx, dy, dz)` versus previous frame.

The extra velocity is intentional: wrist-centering alone removes global motion that is essential for `move` and `wipe`.

## Tests

```bash
pytest -q
```

Unit tests do not require a camera or MediaPipe model file.

## Known limitations
- Skeleton-only `grab` can confuse a plain fist-closing gesture with a true grasp.
- `move`, `wipe`, and `rotate` depend on camera viewpoint and training diversity.
- A 30-frame window adds temporal observation latency.
- V1 tracks one hand only.

## Next upgrade
Add object detection/tracking and hand-object geometric relations so predictions become semantic events such as `GRAB(cloth)` and `WIPE(surface, cloth)`.
