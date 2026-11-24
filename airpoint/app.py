import argparse
import os
import time

import cv2

from . import features as F
from . import hud
from .actions import ActionRunner, PyAutoGuiMouse
from .calibration import CalibSession, Trainer, load_profile, save_profile
from .camera import Camera
from .classifier import PoseClassifier, RuleClassifier, VoteSmoother
from .filters import OneEuroFilter
from .fsm import AirPointFSM, DwellDetector
from .poses import POSES
from .recorder import Recorder, RecordSession, looks_like
from .tracker import HandTracker
from .train import train_personal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR, MODELS_DIR = os.path.join(ROOT, "data"), os.path.join(ROOT, "models")
MODEL_PATH, LABELS_PATH = os.path.join(MODELS_DIR, "pose_mlp.joblib"), os.path.join(MODELS_DIR, "labels.json")
FC_MIN, FC_MIN_PRECLICK, PRECLICK_RATIO = 0.80, 0.30, 0.42
WINDOW, MINI_MARGIN = "AirPoint", 20
KEYS = "q quit | m mouse | d dwell | k rules/mlp | c calibrate | v mini | r record | 1-6 pose | space start/redo"
KEY_HINT = "keys q m d k c v r"


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


def capture(session, recorder, pose, pts, hand, t):
    """Write this frame if the session is capturing and it passes the guard. -> True if rejected."""
    if session.fresh:
        recorder.drop(pose)
        session.fresh = False
    if not session.capturing or pts is None:
        return False
    if looks_like(pose, pts):
        recorder.write(pose, t, hand, pts)
        return False
    return True


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
    clf = mlp or rules
    fc_min = profile["fc_min"] if profile else FC_MIN
    smoother, fsm, dwell = VoteSmoother(), AirPointFSM(), DwellDetector()
    fx, fy = OneEuroFilter(FC_MIN), OneEuroFilter(FC_MIN)
    mouse = PyAutoGuiMouse()
    runner = ActionRunner(mouse, mouse.size())
    if args.mouse:
        runner.toggle()
    dwell_on, mini = args.dwell, False
    recorder = Recorder(os.path.join(DATA_DIR, f"{args.user}.csv"), args.user) if args.record else None
    session = RecordSession() if args.record else None
    calib, calib_rec, trainer = None, None, None
    camera, tracker = Camera(), HandTracker()
    last_event, prev_t, rejected = "none", time.time(), 0
    print(f"[AirPoint] camera {camera.index} | classifier {clf.name} | fc_min {fc_min} | {KEYS}")

    try:
        for frame in camera.frames():
            t = time.time()
            fps, prev_t = 1.0 / max(t - prev_t, 1e-5), t
            h, w = frame.shape[:2]
            pts, hand = tracker.detect(frame)
            raw, mini_label = None, ""
            if pts is not None:
                hud.draw_skeleton(frame, pts)
                raw = (int(pts[F.MIDDLE_MCP, 0] * w), int(pts[F.MIDDLE_MCP, 1] * h))

            if session is not None:
                phase, pose, left = session.update(t)
                lost = capture(session, recorder, pose, pts, hand, t)
                rejected += lost
                title = {
                    "idle": f"RECORD {args.user}: SPACE = all poses, 1-6 = one pose",
                    "countdown": f"raise hand, make {pose}   {left:.0f}",
                    "capturing": f"HOLD {pose}   {left:.0f}s" + ("   << SIGN LOST" if lost else ""),
                    "done": "done - q to quit, 1-6 to redo one pose",
                }[phase]
                hud.draw_prompt(frame, title,
                                f"tilt, move closer, move away | kept {recorder.count} | rejected {rejected} | {KEY_HINT}")
            elif calib is not None:
                phase, pose, left = calib.update(t)
                if phase == "still" and raw is not None:
                    calib.still_points.append(raw)
                lost = capture(calib, calib_rec, pose, pts, hand, t)
                if phase == "training":
                    if trainer is None:
                        calib_rec.close()
                        trainer = Trainer(lambda: train_personal(DATA_DIR, calib_rec.path, MODELS_DIR, args.user))
                        trainer.start()
                    elif trainer.done:
                        if trainer.error:
                            calib.phase, calib.message = "failed", trainer.error
                        else:
                            r = trainer.result
                            mlp = clf = PoseClassifier.load(r["model"], r["labels"])
                            mlp.name = f"mlp/{args.user}"
                            smoother.reset()
                            fc_min = calib.fc_min
                            save_profile(ROOT, args.user, fc_min, r["model"], r["labels"], r["acc_on_own_frames"])
                            calib.phase = "done"
                            calib.message = f"{r['acc_on_own_frames'] * 100:.0f}% on your frames, fc_min {fc_min}"
                        trainer = None
                title = {
                    "countdown": f"CALIBRATE: open palm, hold still   {left:.0f}",
                    "still": f"HOLD STILL   {left:.0f}s",
                    "poses_countdown": f"make {pose}   {left:.0f}",
                    "capturing": f"HOLD {pose}   {left:.0f}s" + ("   << SIGN LOST" if lost else ""),
                    "training": "Learning your hand...",
                    "done": f"done - {calib.message}",
                    "failed": f"retrain failed, old model kept - {calib.message}",
                }[phase]
                hud.draw_prompt(frame, title, f"kept {calib_rec.count} | c = close | {KEY_HINT}")
            else:
                if pts is None:
                    label, conf, ratio = smoother.update(None, 0.0), 0.0, 1.0
                else:
                    raw_label, conf = clf.predict(pts)
                    label, ratio = smoother.update(raw_label, conf), F.pinch_ratio(pts)
                state, events = fsm.update(label, t)
                smooth = None
                if raw is not None:
                    fc = min(FC_MIN_PRECLICK, fc_min) if ratio < PRECLICK_RATIO else fc_min
                    smooth = (int(fx.filter(raw[0], t, fc)), int(fy.filter(raw[1], t, fc)))
                    hud.draw_cursor(frame, raw, smooth)
                    if not fsm.cursor_frozen:
                        runner.move(smooth[0] / w, smooth[1] / h)
                dwell_event, progress = dwell.update(smooth, t, dwell_on and state == "MOVING")
                if dwell_event is not None:
                    events = events + [dwell_event]
                if smooth is not None:
                    hud.draw_dwell_ring(frame, smooth, progress)
                runner.apply(events)
                if events:
                    last_event = ", ".join(e.kind + ("" if e.arg is None else f" {e.arg}") for e in events)
                hud.draw_region(frame, runner.active_region, runner.enabled)
                mini_label = f"{state} | {label} {conf:.2f} | mouse {'ON' if runner.enabled else 'OFF'}"
                if not mini:
                    hud.draw_banner(
                        frame,
                        f"{state} | pose {label} {conf:.2f} | ratio {ratio:.2f} | {clf.name} | {fps:.0f} fps",
                        f"last: {last_event} | mouse [m] {'ON' if runner.enabled else 'OFF'} | "
                        f"dwell [d] {'ON' if dwell_on else 'OFF'} | {KEY_HINT}",
                        runner.enabled,
                    )

            shown = hud.mini_frame(frame, mini_label) if mini and session is None and calib is None else frame
            cv2.imshow(WINDOW, shown)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                runner.toggle()
            elif key == ord("d"):
                dwell_on = not dwell_on
                dwell.reset()
            elif key == ord("k") and mlp is not None:
                clf = rules if clf is mlp else mlp
                smoother.reset()
            elif key == ord("v"):
                mini = not mini
                show_window(mini, runner.screen_w)
            elif key == ord("r") and args.user and calib is None:
                runner.release()
                if session is None:
                    recorder = recorder or Recorder(os.path.join(DATA_DIR, f"{args.user}.csv"), args.user)
                    session = RecordSession()
                else:
                    session = None
            elif key == ord("c") and session is None:
                if not args.user:
                    print("[AirPoint] calibration needs --user <name>")
                elif calib is None:
                    runner.release()
                    calib_rec = Recorder(os.path.join(DATA_DIR, f"{args.user}_calib.csv"), args.user)
                    calib = CalibSession(POSES, t)
                elif calib.phase != "training":
                    calib_rec.close()
                    calib, calib_rec = None, None
            elif session is not None and key == ord(" "):
                session.redo(t)
            elif session is not None and ord("1") <= key <= ord("6"):
                session.select(key - ord("1"), t)
    finally:
        runner.release()
        camera.close()
        tracker.close()
        for r in (recorder, calib_rec):
            if r is not None:
                r.close()
        cv2.destroyAllWindows()
