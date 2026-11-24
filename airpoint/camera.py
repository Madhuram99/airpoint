import cv2


class Camera:
    """Index 1 is the MacBook camera when an iPhone takes index 0; otherwise 0."""

    FPS = 30  # macOS lets the camera idle at 15 fps in dim light unless a rate is requested

    def __init__(self, indices=(1, 0)):
        for i in indices:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FPS, self.FPS)
                if cap.read()[0]:
                    self.cap, self.index = cap, i
                    self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    self.fps = cap.get(cv2.CAP_PROP_FPS)
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
