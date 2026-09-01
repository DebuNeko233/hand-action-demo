# Assembly101 -> MediaPipe -> Hand Action Dataset

This pipeline converts Assembly101 fine-grained action annotations and RGB videos into the same `30 x 66` hand-sequence format used by this project.

Assembly101 fine-grained annotations provide `video`, `start_frame`, `end_frame`, `action_cls`, `verb_cls`, and `noun_cls`. Temporal boundaries are defined at 30 fps.

## Why use verb labels

For an SOP primitive-action library, use `--label-level verb` (default). This merges object-specific actions into reusable primitives. For example, actions such as `pick up bumper` and `pick up screwdriver` share the `pick up` verb.

Object identity should later come from YOLO / hand-object relation logic rather than exploding the motion classifier into hundreds of verb+noun combinations.

## Expected inputs

Download the Assembly101 RGB videos and the official fine-grained annotation CSV files (`train.csv`, `validation.csv`, `test.csv`).

The annotation directory should look like:

```text
fine-grained-annotations/
  actions.csv
  train.csv
  validation.csv
  test.csv
```

The `--video-root` directory must contain the relative video paths referenced by the `video` column in those CSV files, e.g.:

```text
<video-root>/
  nusar-2021_action_both_9033-a30_9033_user_id_2021-02-04_131528/
    C10404_rgb.mp4
```

## 1. Download the MediaPipe hand model

```bash
python tools/download_models.py
```

## 2. Quick smoke test

Start with a small whitelist and only 20 accepted samples per split:

```bash
python tools/prepare_assembly101.py \
  --video-root /path/to/assembly101/videos \
  --annotation-dir /path/to/fine-grained-annotations \
  --labels "pick up,put down,position,screw,unscrew" \
  --limit 20
```

On Windows PowerShell:

```powershell
python tools/prepare_assembly101.py `
  --video-root "D:\Assembly101\videos" `
  --annotation-dir "D:\Assembly101\fine-grained-annotations" `
  --labels "pick up,put down,position,screw,unscrew" `
  --limit 20
```

## 3. Build the full selected dataset

Remove `--limit` after the smoke test succeeds:

```bash
python tools/prepare_assembly101.py \
  --video-root /path/to/assembly101/videos \
  --annotation-dir /path/to/fine-grained-annotations \
  --labels "pick up,put down,position,screw,unscrew"
```

You can also put one verb per line in a text file and pass the file path to `--labels`.

Each accepted segment is processed as:

```text
Assembly101 annotated segment
        |
        v
sample 30 timestamps across the official action interval
        |
        v
MediaPipe Hands (21 x XYZ)
        |
        v
interpolate occasional missed-hand frames
        |
        v
wrist-centered + scale-normalized pose
+ wrist displacement
        |
        v
30 x 66 .npy
```

Segments are discarded when MediaPipe detects a hand in fewer than 60% of sampled frames. Change this with `--min-hand-ratio`.

Generated layout:

```text
dataset_assembly101/
  manifest.json
  metadata.csv
  train/
    pick_up/*.npy
    put_down/*.npy
    ...
  validation/
    ...
  test/
    ...
```

`metadata.csv` retains the source video, temporal bounds, verb/noun/action IDs and hand-detection ratio for every generated sample.

## 4. Train

```bash
python tools/train_assembly101.py
```

Checkpoint:

```text
models/hand_action_assembly101_lstm.pth
```

The training script uses the official Assembly101 train/validation/test splits rather than creating a random split.

## 5. Camera inference

Because both Assembly101 training data and live input are passed through the project's MediaPipe feature extractor, use the normal feature mode:

```bash
python main.py \
  --camera 0 \
  --checkpoint models/hand_action_assembly101_lstm.pth
```

## Important limitation

Assembly101 annotations are trimmed action intervals, while `main.py` currently classifies a rolling 30-frame camera window. For production SOP recognition, add an action-start/action-end segmentation stage or an `idle/background` class so that inference windows resemble the training samples.
