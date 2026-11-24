import argparse
import os
import time

import cv2

from . import hud
from .actions import ActionRunner, PyAutoGuiMouse
from .calibration import load_profile
from .camera import Camera
from .classifier import PoseClassifier, RuleClassifier
from .modes import RecordMode
from .pointer import FC_MIN
from .session import Session
from .tracker import HandTracker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
MODEL_PATH, LABELS_PATH = os.path.join(MODELS_DIR, "pose_mlp.joblib"), os.path.join(MODELS_DIR, "labels.json")
WINDOW, MINI_MARGIN = "AirPoint", 20
KEYS = ("q quit | m mouse | d dwell | b filter full/smooth/off | k rules/mlp | c calibrate | v mini | z zoom | "
        "t simulated tremor ([ ] size, { } speed) | r record | 1-6 pose | space start/redo")


def load_classifiers(force_rules, profile=None):
    """-> (rules, mlp | None). A profile with an existing personal model wins over models/pose_mlp."""
    rules = RuleClassifier()
    model, labels = MODEL_PATH, LABELS_PATH
    if profile is not None and os.path.exists(os.path.join(ROOT, profile["model"])):
        model, labels = os.path.join(ROOT, profile["model"]), os.path.join(ROOT, profile["labels"])
    if force_rules or not os.path.exists(model):
        return rules, None
    mlp = PoseClassifier.load(model, labels)
    if model != MODEL_PATH:
        mlp.name = f"mlp/{profile['user']}"
    return rules, mlp


def show_window(mini, screen_w):
    """Recreate the window; mini = always on top in the top-right corner."""
    try:
        cv2.destroyWindow(WINDOW)
    except cv2.error:
        pass
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    if mini:
        try:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            print("[AirPoint] always-on-top not supported by this OpenCV build")
        cv2.moveWindow(WINDOW, screen_w - hud.MINI_SIZE[0] - MINI_MARGIN, MINI_MARGIN)


def main(argv=None):
    ap = argparse.ArgumentParser(description="AirPoint live demo")
    ap.add_argument("--user", help="your name: loads profiles/<user>.json; enables record (r) and calibrate (c)")
    ap.add_argument("--record", action="store_true", help="start in record mode")
    ap.add_argument("--rules", action="store_true", help="force the rule classifier")
    ap.add_argument("--mouse", action="store_true", help="start with OS mouse control ON")
    ap.add_argument("--dwell", action="store_true", help="start with dwell click ON")
    args = ap.parse_args(argv)
    if args.record and not args.user:
        ap.error("--record needs --user")

    profile = load_profile(ROOT, args.user) if args.user else None
    rules, mlp = load_classifiers(args.rules, profile)
    mouse = PyAutoGuiMouse()
    runner = ActionRunner(mouse, mouse.size())
    if args.mouse:
        runner.toggle()
    s = Session(runner, rules, mlp, root=ROOT, user=args.user,
                fc_min=profile["fc_min"] if profile else FC_MIN, dwell_on=args.dwell)
    if args.record:
        s.set_mode(RecordMode(s))
    camera, tracker = Camera(), HandTracker()
    view_h = round(camera.size[1] * hud.CANON_W / camera.size[0])
    print(f"[AirPoint] camera {camera.index} {camera.size[0]}x{camera.size[1]} @ {camera.fps:.0f} fps -> "
          f"view {hud.CANON_W}x{view_h} | classifier {s.clf.name} | fc_min {s.pointer.fc_min} | {KEYS}")

    try:
        for frame in camera.frames():
            t = time.time()
            pts, hand = tracker.detect(frame)
            shown = s.tick(frame, pts, hand, t)
            if s.view_dirty:
                show_window(s.mini, runner.screen_w)
                s.view_dirty = False
            cv2.imshow(WINDOW, shown)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key != 255:
                s.on_key(key, t)
    finally:
        s.close()
        camera.close()
        tracker.close()
        cv2.destroyAllWindows()
