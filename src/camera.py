from __future__ import annotations
import cv2

class Camera:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open camera index {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        return self.cap.read()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
