from __future__ import annotations
import numpy as np

class StatefulFeatureExtractor:
    """63 wrist-centered landmark values + 3 scale-normalized wrist velocity values."""
    def __init__(self) -> None:
        self.previous_wrist: np.ndarray | None = None

    def reset(self) -> None:
        self.previous_wrist = None

    def extract(self, landmarks: np.ndarray) -> np.ndarray:
        landmarks = np.asarray(landmarks, dtype=np.float32)
        if landmarks.shape != (21, 3):
            raise ValueError(f"Expected landmarks shape (21,3), got {landmarks.shape}")
        if not np.isfinite(landmarks).all():
            raise ValueError("Landmarks contain NaN or Inf")
        wrist = landmarks[0].copy()
        centered = landmarks - wrist
        scale = float(np.linalg.norm(landmarks[9] - wrist))
        scale = max(scale, 1e-6)
        normalized = centered / scale
        if self.previous_wrist is None:
            velocity = np.zeros(3, dtype=np.float32)
        else:
            # Keep the motion term dimensionless. This is important when training
            # from DHG-2016 world-space skeletons and inferring from MediaPipe.
            velocity = (wrist - self.previous_wrist) / scale
        self.previous_wrist = wrist
        feature = np.concatenate([normalized.reshape(-1), velocity], axis=0).astype(np.float32)
        if feature.shape != (66,) or not np.isfinite(feature).all():
            raise ValueError("Invalid feature output")
        return feature
