from __future__ import annotations
from collections import Counter, deque

class PredictionSmoother:
    def __init__(self, window: int = 5, threshold: float = 0.65) -> None:
        self.window = deque(maxlen=window)
        self.threshold = threshold

    def update(self, label: str, confidence: float) -> str:
        value = label if confidence >= self.threshold else "unknown"
        self.window.append(value)
        return Counter(self.window).most_common(1)[0][0]

    def clear(self) -> None:
        self.window.clear()
