from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import HAND_LANDMARKER_MODEL_PATH, SEQUENCE_LENGTH
from src.feature_extractor import StatefulFeatureExtractor
from src.hand_tracker import HandTracker

ANNOTATION_FPS = 30.0
SPLIT_FILES = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def parse_filter(value: str | None) -> set[str] | None:
    if not value:
        return None
    p = Path(value)
    if p.exists():
        items = [line.strip() for line in p.read_text(encoding="utf-8").splitlines()]
    else:
        items = [x.strip() for x in value.split(",")]
    items = [x.lower() for x in items if x]
    return set(items) if items else None


def load_rows(annotation_dir: Path, split: str) -> list[dict[str, str]]:
    path = annotation_dir / SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"Missing Assembly101 annotation file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def interpolate_missing(points: list[np.ndarray | None]) -> np.ndarray | None:
    valid = [i for i, p in enumerate(points) if p is not None]
    if not valid:
        return None
    arr = np.full((len(points), 21, 3), np.nan, dtype=np.float32)
    for i in valid:
        arr[i] = points[i]
    t = np.arange(len(points), dtype=np.float32)
    good_t = np.asarray(valid, dtype=np.float32)
    for joint in range(21):
        for axis in range(3):
            vals = arr[valid, joint, axis]
            arr[:, joint, axis] = np.interp(t, good_t, vals)
    return arr


def extract_segment(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    tracker: HandTracker,
    sequence_length: int,
    min_hand_ratio: float,
) -> tuple[np.ndarray | None, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0.0

    start_sec = max(0.0, float(start_frame) / ANNOTATION_FPS)
    end_sec = max(start_sec, float(end_frame) / ANNOTATION_FPS)
    if end_sec <= start_sec:
        cap.release()
        return None, 0.0

    times = np.linspace(start_sec, end_sec, sequence_length, endpoint=False, dtype=np.float64)
    landmarks: list[np.ndarray | None] = []
    try:
        for sec in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(sec * 1000.0))
            ok, frame = cap.read()
            if not ok:
                landmarks.append(None)
                continue
            result = tracker.process(frame)
            landmarks.append(result.landmarks.copy() if result.detected and result.landmarks is not None else None)
    finally:
        cap.release()

    valid_ratio = sum(p is not None for p in landmarks) / max(1, len(landmarks))
    if valid_ratio < min_hand_ratio:
        return None, valid_ratio

    filled = interpolate_missing(landmarks)
    if filled is None:
        return None, valid_ratio

    extractor = StatefulFeatureExtractor()
    features = np.stack([extractor.extract(frame) for frame in filled]).astype(np.float32)
    return features, valid_ratio


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Assembly101 fine-grained videos -> MediaPipe 21-point hand skeleton -> fixed 30x66 training samples."
    )
    ap.add_argument("--video-root", type=Path, required=True,
                    help="Root directory containing the sequence/view paths referenced by Assembly101 CSV files.")
    ap.add_argument("--annotation-dir", type=Path, required=True,
                    help="Directory containing official fine-grained train.csv, validation.csv and test.csv.")
    ap.add_argument("--output", type=Path, default=ROOT / "dataset_assembly101")
    ap.add_argument("--label-level", choices=["verb", "action"], default="verb",
                    help="Use generic verb labels (recommended for SOP primitives) or full verb+noun action labels.")
    ap.add_argument("--labels", type=str, default=None,
                    help="Optional comma-separated whitelist or text file. Example: 'pick up,put down,screw,unscrew,position'.")
    ap.add_argument("--splits", nargs="+", choices=list(SPLIT_FILES), default=list(SPLIT_FILES))
    ap.add_argument("--min-hand-ratio", type=float, default=0.60,
                    help="Skip a segment when MediaPipe detects a hand in less than this fraction of sampled frames.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Optional maximum number of accepted samples per split for quick tests; 0 means unlimited.")
    args = ap.parse_args()

    if not HAND_LANDMARKER_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing MediaPipe hand model: {HAND_LANDMARKER_MODEL_PATH}. Run python tools/download_models.py first."
        )

    label_filter = parse_filter(args.labels)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "metadata.csv"
    metadata_exists = metadata_path.exists()

    # Offline extraction samples independent frames from annotated intervals.
    # IMAGE mode avoids unnecessary temporal graph state and is safer on macOS.
    tracker = HandTracker(HAND_LANDMARKER_MODEL_PATH, max_num_hands=1, running_mode="image")
    class_names: set[str] = set()
    stats: dict[str, dict[str, int]] = {}

    try:
        with metadata_path.open("a", encoding="utf-8", newline="") as meta_f:
            writer = csv.DictWriter(meta_f, fieldnames=[
                "split", "sample", "label", "label_text", "video", "start_frame", "end_frame",
                "action_id", "verb_id", "noun_id", "action_cls", "verb_cls", "noun_cls", "hand_ratio",
            ])
            if not metadata_exists:
                writer.writeheader()

            for split in args.splits:
                rows = load_rows(args.annotation_dir, split)
                accepted = skipped_filter = skipped_missing_video = skipped_hand = 0
                for row in tqdm(rows, desc=f"Assembly101 {split}"):
                    label_text = row["verb_cls"] if args.label_level == "verb" else row["action_cls"]
                    if label_filter is not None and label_text.strip().lower() not in label_filter:
                        skipped_filter += 1
                        continue

                    video_path = args.video_root / row["video"]
                    if not video_path.exists():
                        skipped_missing_video += 1
                        continue

                    label = slugify(label_text)
                    features, hand_ratio = extract_segment(
                        video_path=video_path,
                        start_frame=int(row["start_frame"]),
                        end_frame=int(row["end_frame"]),
                        tracker=tracker,
                        sequence_length=SEQUENCE_LENGTH,
                        min_hand_ratio=args.min_hand_ratio,
                    )
                    if features is None:
                        skipped_hand += 1
                        continue

                    out_dir = args.output / split / label
                    out_dir.mkdir(parents=True, exist_ok=True)
                    sample_id = f"{split}_{int(row['id']):08d}"
                    out_path = out_dir / f"{sample_id}.npy"
                    np.save(out_path, features)
                    class_names.add(label)
                    writer.writerow({
                        "split": split,
                        "sample": str(out_path.relative_to(args.output)),
                        "label": label,
                        "label_text": label_text,
                        "video": row["video"],
                        "start_frame": row["start_frame"],
                        "end_frame": row["end_frame"],
                        "action_id": row.get("action_id", ""),
                        "verb_id": row.get("verb_id", ""),
                        "noun_id": row.get("noun_id", ""),
                        "action_cls": row.get("action_cls", ""),
                        "verb_cls": row.get("verb_cls", ""),
                        "noun_cls": row.get("noun_cls", ""),
                        "hand_ratio": f"{hand_ratio:.3f}",
                    })
                    meta_f.flush()
                    accepted += 1
                    if args.limit > 0 and accepted >= args.limit:
                        break

                stats[split] = {
                    "accepted": accepted,
                    "skipped_filter": skipped_filter,
                    "skipped_missing_video": skipped_missing_video,
                    "skipped_hand": skipped_hand,
                }
    finally:
        tracker.close()

    manifest = {
        "dataset": "Assembly101",
        "annotation_fps": ANNOTATION_FPS,
        "sequence_length": SEQUENCE_LENGTH,
        "input_size": 66,
        "label_level": args.label_level,
        "class_names": sorted(class_names),
        "stats": stats,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
