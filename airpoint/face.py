"""Face as the input: head direction moves the pointer, expressions are the signs.

Everything after the tracker is shared with the hand: vote, state machine, filters, dwell, wheel.
Angles are degrees from MediaPipe's head-pose matrix, in the mirrored view the app shows:
a larger yaw = head turned toward the right of the picture, a larger pitch = looking up.
"""
import math
import os
from dataclasses import dataclass, field

import numpy as np

from .fsm import Event

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "face_landmarker.task")
DOWNLOAD = ("curl -L -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

NEUTRAL, MOUTH_OPEN, BROWS_UP = "neutral", "mouth_open", "brows_up"
FACE_LABELS = (NEUTRAL, MOUTH_OPEN, BROWS_UP)
SCORE_OF = {MOUTH_OPEN: "jawOpen", BROWS_UP: "browInnerUp"}        # MediaPipe expression score behind each sign
GRIP, HOLD_EVENTS = frozenset({MOUTH_OPEN}), {BROWS_UP: Event("right_click")}
OFF_SHARE = 0.6            # a sign that is on stays on until its score falls below this share of the threshold
NOSE, FOREHEAD, CHIN, SIDE_A, SIDE_B, LIP_TOP, LIP_BOTTOM = 1, 10, 152, 234, 454, 13, 14
MIN_RISE, SPAN_FLOOR = 0.15, (5.0, 4.0)


def default_face():
    """What face mode uses before calibration: resting pose straight ahead, a modest comfortable range."""
    return {"yaw0": 0.0, "pitch0": 0.0, "span_yaw": 18.0, "span_pitch": 12.0,
            "on": {MOUTH_OPEN: 0.45, BROWS_UP: 0.45}}


def head_angles(matrix):
    """4x4 head-pose matrix -> (yaw, pitch) in degrees: where the face's forward axis points."""
    fwd = np.asarray(matrix, dtype=float)[:3, :3] @ np.array([0.0, 0.0, 1.0])
    return math.degrees(math.atan2(fwd[0], fwd[2])), math.degrees(math.atan2(fwd[1], fwd[2]))


@dataclass
class FaceObs:
    """One frame of a face."""
    yaw: float
    pitch: float
    shapes: dict                                   # expression name -> score 0..1
    marks: dict = field(default_factory=dict)      # a few landmarks (normalized x, y) for drawing


class FacePointer:
    """The comfortable head range mapped onto the whole view; the resting pose is the centre."""

    def __init__(self, cfg):
        self.cfg = cfg

    def xy(self, yaw, pitch, w, h):
        c = self.cfg
        x = 0.5 + (yaw - c["yaw0"]) / (2.0 * c["span_yaw"])
        y = 0.5 - (pitch - c["pitch0"]) / (2.0 * c["span_pitch"])
        return (min(max(x, 0.0), 1.0) * w, min(max(y, 0.0), 1.0) * h)


class FaceSigns:
    """Expression scores -> one label, with hysteresis so a score hovering at its threshold cannot chatter."""

    def __init__(self, cfg):
        self.on, self.active = dict(cfg["on"]), None

    def update(self, shapes):
        """-> (label, probabilities in FACE_LABELS order, how close the mouth is to clicking 0..1)."""
        level = {label: (shapes.get(SCORE_OF[label], 0.0) / self.on[label] if label in self.on else 0.0)
                 for label in (MOUTH_OPEN, BROWS_UP)}
        if self.active is not None and level[self.active] < OFF_SHARE:
            self.active = None
        if self.active is None:
            best = max(level, key=level.get)
            if level[best] >= 1.0:
                self.active = best
        probs = np.array([0.0] + [min(level[MOUTH_OPEN], 1.0), min(level[BROWS_UP], 1.0)])
        probs[0] = max(1.0 - probs[1:].max(), 0.0)
        return self.active or NEUTRAL, probs / probs.sum(), min(level[MOUTH_OPEN], 1.0)


class FaceCalib:
    """Timing and arithmetic of face calibration: rest -> reach -> open mouth -> raise brows."""

    PHASES = (("rest", 3.0), ("reach", 6.0), ("mouth", 3.0), ("brows", 3.0))

    def __init__(self, t):
        self.t0, self.samples = t, {name: [] for name, _ in self.PHASES}
        self.result, self.dropped = None, []

    def phase(self, t):
        """-> (name, seconds left); name is 'done' at the end."""
        elapsed = t - self.t0
        for name, seconds in self.PHASES:
            if elapsed < seconds:
                return name, seconds - elapsed
            elapsed -= seconds
        return "done", 0.0

    def add(self, t, obs):
        name, _ = self.phase(t)
        if name != "done" and obs is not None:
            self.samples[name].append((obs.yaw, obs.pitch, obs.shapes.get("jawOpen", 0.0), obs.shapes.get("browInnerUp", 0.0)))

    def finish(self):
        """-> the profile's `face` settings, or None if the face was not seen enough."""
        s = {name: np.array(rows, dtype=float).reshape(-1, 4) for name, rows in self.samples.items()}
        if len(s["rest"]) < 20 or len(s["reach"]) < 40:
            return None
        yaw0, pitch0 = float(np.median(s["rest"][:, 0])), float(np.median(s["rest"][:, 1]))
        spans = []
        for col, centre, floor in ((0, yaw0, SPAN_FLOOR[0]), (1, pitch0, SPAN_FLOOR[1])):
            lo, hi = np.percentile(s["reach"][:, col], [5, 95])
            spans.append(max(min(hi - centre, centre - lo), floor))       # the smaller side: both edges reachable
        quiet = np.concatenate([s["rest"], s["reach"]])
        on, self.dropped = {}, []
        for label, col, phase in ((MOUTH_OPEN, 2, "mouth"), (BROWS_UP, 3, "brows")):
            noise = float(np.percentile(quiet[:, col], 95))
            peak = float(np.percentile(s[phase][:, col], 90)) if len(s[phase]) else 0.0
            if peak - noise < MIN_RISE:
                self.dropped.append(label)          # this face could not make it, or it could not be seen
            else:
                on[label] = round(noise + 0.5 * (peak - noise), 3)
        self.result = {"yaw0": round(yaw0, 2), "pitch0": round(pitch0, 2), "span_yaw": round(float(spans[0]), 2),
                       "span_pitch": round(float(spans[1]), 2), "on": on}
        return self.result


class FaceTracker:
    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} missing. Download with:\n  {DOWNLOAD}")
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE, num_faces=1,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True)
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr):
        """-> FaceObs, or None when no face is in view."""
        import cv2
        import mediapipe as mp
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        r = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not r.face_landmarks or not r.facial_transformation_matrixes:
            return None
        lm = r.face_landmarks[0]
        yaw, pitch = head_angles(r.facial_transformation_matrixes[0])
        shapes = {c.category_name: float(c.score) for c in r.face_blendshapes[0]} if r.face_blendshapes else {}
        marks = {name: (lm[i].x, lm[i].y) for name, i in (("nose", NOSE), ("forehead", FOREHEAD), ("chin", CHIN),
                                                         ("side_a", SIDE_A), ("side_b", SIDE_B),
                                                         ("lip_top", LIP_TOP), ("lip_bottom", LIP_BOTTOM))}
        return FaceObs(yaw, pitch, shapes, marks)

    def close(self):
        self.detector.close()
