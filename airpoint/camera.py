import cv2


class Camera:
    """Index 1 is the MacBook camera when an iPhone takes index 0; otherwise 0."""

    def __init__(self, indices=(1, 0)):
        for i in indices:
            cap = cv2.VideoCapture(i)
            if cap.isOpened() and cap.read()[0]:
                self.cap, self.index = cap, i
                return
            cap.release()
        raise RuntimeError("Cannot open webcam. System Settings > Privacy & Security > Camera.")

    def frames(self):
        while self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                return
            yield cv2.flip(frame, 1)

    def close(self):
        self.cap.release()
