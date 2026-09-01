from __future__ import annotations
import numpy as np

FINGERTIPS = [4, 8, 12, 16, 20]
MCPS = [2, 5, 9, 13, 17]

def compute_heuristics(landmarks: np.ndarray, previous_wrist: np.ndarray | None = None) -> dict[str, float]:
    wrist = landmarks[0]
    openness = float(np.mean([np.linalg.norm(landmarks[t] - landmarks[m]) for t, m in zip(FINGERTIPS, MCPS)]))
    if previous_wrist is None:
        velocity = np.zeros(3, dtype=np.float32)
    else:
        velocity = wrist - previous_wrist
    return {
        "hand_openness": openness,
        "wrist_dx": float(velocity[0]),
        "wrist_dy": float(velocity[1]),
        "wrist_dz": float(velocity[2]),
        "wrist_speed": float(np.linalg.norm(velocity)),
    }
