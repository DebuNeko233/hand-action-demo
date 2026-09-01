from __future__ import annotations
from pathlib import Path
from urllib.request import urlretrieve
import sys

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models" / "hand_landmarker.task"
# MediaPipe official model asset URL used by the Hand Landmarker guide.
URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}\n -> {DEST}")
    try:
        urlretrieve(URL, DEST)
    except Exception as exc:
        if DEST.exists(): DEST.unlink()
        print(f"Download failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Done ({DEST.stat().st_size/1024/1024:.1f} MB)")

if __name__ == "__main__": main()
