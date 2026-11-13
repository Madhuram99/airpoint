import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "hand_landmarker.task")
DOWNLOAD = ("curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")


class HandTracker:
    def __init__(self, model_path=MODEL_PATH, num_hands=1):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} missing. Download with:\n  {DOWNLOAD}")
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr):
        """-> ((21, 3) array in normalized image coords, handedness) or (None, None)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.hand_landmarks:
            return None, None
        pts = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]])
        hand = result.handedness[0][0].category_name if result.handedness else "Unknown"
        return pts, hand

    def close(self):
        self.detector.close()
