import argparse
import os
import time

import cv2

from . import hud, ui
from .actions import ActionRunner, PyAutoGuiMouse
from . import profile as profiles
from .camera import Camera
from .classifier import PoseClassifier, RuleClassifier
from .face import FaceTracker
from .helper import CONSENT
from .modes import RecordMode
from .overlay import Overlay
from .session import Session
from .tracker import HandTracker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
MODEL_PATH, LABELS_PATH = os.path.join(MODELS_DIR, "pose_mlp.joblib"), os.path.join(MODELS_DIR, "labels.json")
WINDOW, MINI_MARGIN = "AirPoint", 20
DRAIN_FRAMES = 5     # after AirPoint types, this many frames without a key before keys are commands again
KEYS = ("q quit | m mouse | d dwell | w wheel | b filter full/smooth/off | k rules/mlp | c calibrate | f face/hand | v mini | z zoom | x x-ray | o cursor glow | "
        "t simulated tremor ([ ] size, { } speed) | p pause | l talk | a AI helper | h hints | s sounds | n new sign | r record | 1-6 pose | "
        "space start/redo | ? all keys")


def load_classifiers(force_rules, profile=None):
    """-> (rules, mlp | None). A profile with an existing personal model wins over models/pose_mlp."""
    rules = RuleClassifier()
    model, labels = MODEL_PATH, LABELS_PATH
    if profile is not None and profile.model and os.path.exists(os.path.join(ROOT, profile.model)):
        model, labels = os.path.join(ROOT, profile.model), os.path.join(ROOT, profile.labels)
    if force_rules or not os.path.exists(model):
        return rules, None
    mlp = PoseClassifier.load(model, labels)
    if model != MODEL_PATH:
        mlp.name = f"mlp/{profile.user}"
    return rules, mlp


def fit_window(shape, backing, mini, screen_w):
    """Size the window in points for a picture drawn in real pixels: on a Retina display the picture has
    twice as many pixels as the window has points, which is what makes its text sharp. The small view
    also stays above other windows, in the top-right corner."""
    w, h = max(round(shape[1] / backing), 1), max(round(shape[0] / backing), 1)
    cv2.resizeWindow(WINDOW, w, h)
    try:
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1 if mini else 0)
    except cv2.error:
        if mini:
            print("[AirPoint] always-on-top not supported by this OpenCV build")
    if mini:
        cv2.moveWindow(WINDOW, screen_w - w - MINI_MARGIN, MINI_MARGIN)


def main(argv=None):
    ap = argparse.ArgumentParser(description="AirPoint live demo")
    ap.add_argument("--user", help="your name: loads profiles/<user>.json; enables record (r) and calibrate (c)")
    ap.add_argument("--record", action="store_true", help="start in record mode")
    ap.add_argument("--rules", action="store_true", help="force the rule classifier")
    ap.add_argument("--mouse", action="store_true", help="start with OS mouse control ON")
    ap.add_argument("--dwell", action="store_true", help="start with dwell click ON")
    ap.add_argument("--no-overlay", action="store_true", help="no glow at the cursor (the overlay is macOS only)")
    ap.add_argument("--theme", choices=sorted(ui.THEMES), help="window colours; default: follow the system's light or dark look")
    ap.add_argument("--helper", action="store_true",
                    help=f"start with the AI helper switched on; it {CONSENT}")
    args = ap.parse_args(argv)
    if args.record and not args.user:
        ap.error("--record needs --user")

    profile = profiles.load(ROOT, args.user) if args.user else None
    rules, mlp = load_classifiers(args.rules, profile)
    mouse = PyAutoGuiMouse()
    runner = ActionRunner(mouse, mouse.size())
    if args.mouse:
        runner.toggle()
    theme, backing = ui.THEMES[args.theme] if args.theme else ui.system_theme(), ui.backing_scale()
    s = Session(runner, rules, mlp, root=ROOT, user=args.user, profile=profile,
                dwell_on=True if args.dwell else None, overlay=None if args.no_overlay else Overlay(),
                theme=theme, backing=backing)
    if args.record:
        s.set_mode(RecordMode(s))
    if args.helper:
        problem = s.helper.problem()
        s.helper.armed = problem is None
        print(f"[AirPoint] AI helper: {problem}" if problem else f"[AirPoint] AI helper is ON: it {CONSENT}")
    camera, tracker, face_tracker = Camera(), HandTracker(), None
    view_h = round(camera.size[1] * hud.CANON_W / camera.size[0])
    print(f"[AirPoint] camera {camera.index} {camera.size[0]}x{camera.size[1]} @ {camera.fps:.0f} fps -> "
          f"view {hud.CANON_W}x{view_h} | classifier {s.clf.name} | fc_min {s.pointer.fc_min} | "
          f"talk: {'on this laptop' if s.talker.local else 'macOS dictation'} | {KEYS}")

    draining, shape = 0, None
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    try:
        for frame in camera.frames():
            t = time.time()
            t0 = time.perf_counter()
            pts = hand = face = None
            if s.source == "face" and face_tracker is None:
                try:
                    face_tracker = FaceTracker()
                except FileNotFoundError as e:
                    print(f"[AirPoint] face mode needs its model. {e}")
                    s.set_source("hand")
            if s.source == "face":
                face = face_tracker.detect(frame)
            else:
                pts, hand = tracker.detect(frame)
            shown = s.tick(frame, pts, hand, t, tracker_ms=(time.perf_counter() - t0) * 1000.0, face=face)
            cv2.imshow(WINDOW, shown)
            key = cv2.waitKey(1) & 0xFF
            if shown.shape[:2] != shape or s.view_dirty:      # a new size (zoom, x-ray, small view): fit the window to it
                shape, s.view_dirty = shown.shape[:2], False
                fit_window(shape, backing, s.mini, runner.screen_w)
            if runner.typing:
                draining = DRAIN_FRAMES
            if draining:                 # with this window in front, what AirPoint types comes back here: not commands
                draining = DRAIN_FRAMES if key != 255 else draining - 1
                key = 255
            if key == ord("q") and not s.typing:
                break
            if key != 255:
                s.on_key(key, t)
    finally:
        s.close()
        camera.close()
        tracker.close()
        if face_tracker is not None:
            face_tracker.close()
        cv2.destroyAllWindows()
