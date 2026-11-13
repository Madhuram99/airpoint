import argparse
import os
import time

import cv2

from . import features as F
from . import hud
from .actions import ActionRunner, PyAutoGuiMouse
from .camera import Camera
from .classifier import PoseClassifier, RuleClassifier, VoteSmoother
from .filters import OneEuroFilter
from .fsm import AirPointFSM
from .recorder import Recorder, RecordSession
from .tracker import HandTracker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "pose_mlp.joblib")
LABELS_PATH = os.path.join(ROOT, "models", "labels.json")
FC_MIN, FC_MIN_PRECLICK, PRECLICK_RATIO = 0.80, 0.30, 0.42
WINDOW = "AirPoint"
KEYS = "q quit | m mouse | k rules/mlp | r record mode | 1-6 pick pose | space start/redo"


def load_classifiers(force_rules):
    rules = RuleClassifier()
    if force_rules or not os.path.exists(MODEL_PATH):
        return rules, None
    return rules, PoseClassifier.load(MODEL_PATH, LABELS_PATH)


def main(argv=None):
    ap = argparse.ArgumentParser(description="AirPoint live demo")
    ap.add_argument("--user", help="name for data/<user>.csv; enables record mode")
    ap.add_argument("--record", action="store_true", help="start in record mode")
    ap.add_argument("--rules", action="store_true", help="force the rule classifier")
    args = ap.parse_args(argv)
    if args.record and not args.user:
        ap.error("--record needs --user")

    rules, mlp = load_classifiers(args.rules)
    clf = mlp or rules
    smoother, fsm = VoteSmoother(), AirPointFSM()
    fx, fy = OneEuroFilter(FC_MIN), OneEuroFilter(FC_MIN)
    mouse = PyAutoGuiMouse()
    runner = ActionRunner(mouse, mouse.size())
    recorder = Recorder(os.path.join(ROOT, "data", f"{args.user}.csv"), args.user) if args.user else None
    session = RecordSession() if args.record else None
    camera, tracker = Camera(), HandTracker()
    last_event, prev_t = "none", time.time()
    print(f"[AirPoint] camera {camera.index} | classifier {clf.name} | {KEYS}")

    try:
        for frame in camera.frames():
            t = time.time()
            fps, prev_t = 1.0 / max(t - prev_t, 1e-5), t
            h, w = frame.shape[:2]
            pts, hand = tracker.detect(frame)
            if pts is not None:
                hud.draw_skeleton(frame, pts)

            if session is not None:
                phase, pose, left = session.update(t)
                if session.capturing and pts is not None:
                    recorder.write(pose, t, hand, pts)
                title = {
                    "idle": f"RECORD {args.user}: press SPACE to start",
                    "countdown": f"get ready: {pose}   {left:.0f}",
                    "capturing": f"HOLD {pose}   {left:.0f}s",
                    "done": "done - press q, or 1-6 to redo a pose",
                }[phase]
                hud.draw_prompt(frame, title, f"tilt, move closer, move away | rows {recorder.count} | {KEYS}")
            else:
                if pts is None:
                    label, conf, ratio = smoother.update(None, 0.0), 0.0, 1.0
                else:
                    raw_label, conf = clf.predict(pts)
                    label, ratio = smoother.update(raw_label, conf), F.pinch_ratio(pts)
                state, events = fsm.update(label, t)
                if pts is not None:
                    rx, ry = int(pts[F.MIDDLE_MCP, 0] * w), int(pts[F.MIDDLE_MCP, 1] * h)
                    fc = FC_MIN_PRECLICK if ratio < PRECLICK_RATIO else FC_MIN
                    sx, sy = int(fx.filter(rx, t, fc)), int(fy.filter(ry, t, fc))
                    hud.draw_cursor(frame, (rx, ry), (sx, sy))
                    if not fsm.cursor_frozen:
                        runner.move(sx / w, sy / h)
                runner.apply(events)
                if events:
                    last_event = ", ".join(e.kind + ("" if e.arg is None else f" {e.arg}") for e in events)
                hud.draw_region(frame, runner.active_region, runner.enabled)
                hud.draw_banner(
                    frame,
                    f"{state} | pose {label} {conf:.2f} | ratio {ratio:.2f} | {clf.name} | {fps:.0f} fps",
                    f"last: {last_event} | mouse [m] {'ON' if runner.enabled else 'OFF'} | {KEYS}",
                    runner.enabled,
                )

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                runner.toggle()
            elif key == ord("k") and mlp is not None:
                clf = rules if clf is mlp else mlp
                smoother.reset()
            elif key == ord("r") and recorder is not None:
                runner.release()
                session = None if session is not None else RecordSession()
            elif session is not None and key == ord(" "):
                session.redo(t)
            elif session is not None and ord("1") <= key <= ord("6"):
                session.select(key - ord("1"), t)
    finally:
        runner.release()
        camera.close()
        tracker.close()
        if recorder is not None:
            recorder.close()
        cv2.destroyAllWindows()
