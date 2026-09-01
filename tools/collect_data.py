from __future__ import annotations
import argparse, sys
from pathlib import Path
import cv2
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from config import *
from src.camera import Camera
from src.feature_extractor import StatefulFeatureExtractor
from src.hand_tracker import HandTracker
from src.sequence_buffer import SequenceBuffer
from src.utils import FPSMeter

def next_path(folder: Path) -> Path:
    nums = [int(p.stem) for p in folder.glob("*.npy") if p.stem.isdigit()]
    return folder / f"{(max(nums, default=0)+1):06d}.npy"

def main() -> None:
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", required=True, choices=CLASS_NAMES)
    ap.add_argument("--camera", type=int, default=CAMERA_INDEX)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--stride", type=int, default=30)
    args = ap.parse_args()
    folder = DATASET_DIR / args.action; folder.mkdir(parents=True, exist_ok=True)
    camera = Camera(args.camera, CAMERA_WIDTH, CAMERA_HEIGHT)
    tracker = HandTracker(HAND_LANDMARKER_MODEL_PATH, MAX_NUM_HANDS)
    extractor = StatefulFeatureExtractor(); buffer = SequenceBuffer(SEQUENCE_LENGTH, INPUT_SIZE); fpsm = FPSMeter()
    recording, count, since_save = False, 0, 999
    try:
        while count < args.samples:
            ok, frame = camera.read()
            if not ok: break
            res = tracker.process(frame)
            if res.detected and res.landmarks is not None:
                tracker.draw(frame, res.landmarks)
                if recording:
                    buffer.append(extractor.extract(res.landmarks)); since_save += 1
                    if args.auto and buffer.is_ready() and since_save >= args.stride:
                        np.save(next_path(folder), buffer.get_sequence()); count += 1; since_save = 0
            cv2.putText(frame, f"Action: {args.action}  Samples: {count}/{args.samples}", (20,35), cv2.FONT_HERSHEY_SIMPLEX, .75, (255,255,255), 2)
            cv2.putText(frame, f"Recording: {'YES' if recording else 'NO'}  Buffer: {len(buffer)}/{SEQUENCE_LENGTH}  FPS:{fpsm.tick():.1f}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
            cv2.imshow("Collect Data", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '): recording = not recording
            elif key in (ord('s'),ord('S')) and buffer.is_ready(): np.save(next_path(folder), buffer.get_sequence()); count += 1; since_save = 0
            elif key in (ord('r'),ord('R')): buffer.clear(); extractor.reset()
            elif key in (ord('q'),ord('Q')): break
    finally:
        tracker.close(); camera.release(); cv2.destroyAllWindows()

if __name__ == "__main__": main()
