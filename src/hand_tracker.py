from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

HAND_CONNECTIONS = (
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
)

@dataclass
class HandResult:
    detected: bool
    landmarks: Optional[np.ndarray] = None
    handedness: Optional[str] = None

class HandTracker:
    """MediaPipe Tasks HandLandmarker wrapper using VIDEO mode."""
    def __init__(self, model_path: Path, max_num_hands: int = 1) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing MediaPipe model: {model_path}. Run: python tools/download_models.py"
            )
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("mediapipe is not installed. Run pip install -r requirements.txt") from exc

        self.mp = mp
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self.timestamp_ms = 0

    def process(self, frame_bgr: np.ndarray) -> HandResult:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        self.timestamp_ms += 1
        result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
        if not result.hand_landmarks:
            return HandResult(False)
        pts = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)
        handedness = None
        if result.handedness and result.handedness[0]:
            handedness = result.handedness[0][0].category_name
        return HandResult(True, pts, handedness)

    def close(self) -> None:
        self.landmarker.close()

    @staticmethod
    def draw(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        pts = [(int(x*w), int(y*h)) for x, y, _ in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for p in pts:
            cv2.circle(frame, p, 4, (0, 0, 255), -1)
        return frame
