from __future__ import annotations
from collections import deque
import time

class FPSMeter:
    def __init__(self, window: int = 30) -> None:
        self.times = deque(maxlen=window)
        self.last = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        self.times.append(now - self.last)
        self.last = now
        mean = sum(self.times) / len(self.times)
        return 1.0 / mean if mean > 0 else 0.0
