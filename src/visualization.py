from __future__ import annotations
import cv2

def draw_status(frame, action: str, confidence: float, fps: float, buffer_size: int, buffer_capacity: int, raw_action: str | None = None):
    lines = [
        f"Action: {action.upper()}",
        f"Confidence: {confidence:.2f}",
        f"Buffer: {buffer_size}/{buffer_capacity}",
        f"FPS: {fps:.1f}",
    ]
    if raw_action is not None:
        lines.append(f"Raw: {raw_action}")
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (20, 35 + 30*i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)
    return frame
