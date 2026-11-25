"""What the browser demo needs from the Python side: the pose network written out as numbers, and a
parity fixture (inputs, with the answers Python gives) that keeps try/airpoint.js honest.

    venv/bin/python -m airpoint.export_web          # writes try/model.json and try/parity.json
    node try/parity.mjs                             # the JavaScript port must give the same answers
"""
import csv
import json
import os
import tempfile

import numpy as np

from . import features as F
from .actions import ACTIVE_REGION, ActionRunner, FakeMouse
from .classifier import PoseClassifier, RuleClassifier, VoteSmoother
from .effects import Effects
from .filters import OneEuroFilter
from .fsm import AirPointFSM, DwellDetector
from .helper import Helper
from .lab.stats import block_summary, sequence_stats
from .pointer import BETA, FC_MIN, FC_MIN_PRECLICK, PRECLICK_RATIO, PointerPipeline
from .poses import NONE, UNKNOWN
from .session import Session
from .sounds import Sounds
from .speech import Talker

DIGITS = 7                  # weights are written to float32 precision; the fixture is computed from what was written
VIEW = (640, 360)           # the canonical view the pointer lives in
SCREEN = (1000, 500)        # the pretend screen of the whole-session script
DT = 1.0 / 30.0


def export_model(clf, labels):
    """sklearn MLPClassifier -> {"labels", "layers": [{"w": rows of (inputs x outputs), "b"}]}."""
    cut = lambda v: float(f"{float(v):.{DIGITS}g}")
    return {"labels": [str(x) for x in labels],
            "layers": [{"w": [[cut(v) for v in row] for row in w], "b": [cut(v) for v in b]}
                       for w, b in zip(clf.coefs_, clf.intercepts_)]}


def forward(model, x):
    """The exported network on one feature vector -> class probabilities (relu layers, softmax out)."""
    a = np.asarray(x, dtype=np.float64)
    for i, layer in enumerate(model["layers"]):
        a = a @ np.asarray(layer["w"]) + np.asarray(layer["b"])
        if i < len(model["layers"]) - 1:
            a = np.maximum(a, 0.0)
    e = np.exp(a - a.max())
    return e / e.sum()


class Exported:
    """The exported numbers behind the classifier interface, so the real Session can run on them."""

    name = "mlp"

    def __init__(self, model):
        self.model, self.labels = model, model["labels"]

    def proba(self, landmarks):
        return forward(self.model, F.normalize_63(landmarks))


def read_hands(path):
    """data/<user>.csv -> {pose: [(21, 3) arrays]}."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            pts = np.array([[float(row[f"{a}{i}"]) for a in "xyz"] for i in range(21)])
            out.setdefault(row["pose"], []).append(pts)
    return out


def _hands(model, hands, per_pose=5):
    rows = []
    for pose in sorted(hands):
        pick = hands[pose][::max(len(hands[pose]) // per_pose, 1)][:per_pose]
        for pts in pick:
            p = forward(model, F.normalize_63(pts))
            rows.append({"pts": pts.tolist(), "features": F.normalize_63(pts).tolist(), "pinch": F.pinch_ratio(pts),
                         "palm_px": F.palm_px(pts, *VIEW), "proba": p.tolist(), "label": model["labels"][int(np.argmax(p))]})
    return rows


def _vote(rng):
    steps, v, out = [], VoteSmoother(), []
    for _ in range(60):
        label = [None, "palm", "palm", "pinch", "fist"][int(rng.integers(0, 5))]
        conf = round(float(rng.uniform(0.5, 1.0)), 3)
        steps.append([label, conf])
        out.append(v.update(label, conf))
    return {"steps": steps, "out": out}


def _filter(rng):
    f, t, steps, out = OneEuroFilter(FC_MIN, BETA), 0.0, [], []
    for i in range(120):
        t += DT * float(rng.uniform(0.7, 1.4))
        x = 300.0 + 120.0 * np.sin(i / 17.0) + float(rng.normal(0, 3.0))
        fc = FC_MIN_PRECLICK if 40 <= i < 60 else None
        steps.append([x, t, fc])
        out.append(f.filter(x, t, fc))
    return {"fc_min": FC_MIN, "beta": BETA, "steps": steps, "out": out}


def _pointer(rng):
    p, t, steps, out = PointerPipeline(FC_MIN), 0.0, [], []
    p.mode = "smooth"
    for i in range(90):
        t += DT
        raw = (320.0 + 150.0 * np.cos(i / 11.0) + float(rng.normal(0, 2.0)), 180.0 + 80.0 * np.sin(i / 7.0) + float(rng.normal(0, 2.0)))
        pre = 30 <= i < 45
        steps.append([list(raw), t, pre])
        out.append(list(p.update(raw, t, preclick=pre)))
    return {"steps": steps, "out": out}


def _fsm():
    script = ([("palm", 10), ("pinch", 5), ("palm", 6), ("fist", 20), ("palm", 5), ("v_sign", 14), ("palm", 3), ("v_sign", 12),
               (UNKNOWN, 3), ("thumb_up", 10), ("thumb_down", 8), ("palm", 4), ("pinch", 20), (NONE, 4), ("palm", 3),
               ("v_sign", 4), ("pinch", 3), ("palm", 3)])
    fsm, t, steps, out = AirPointFSM(), 0.0, [], []
    for pose, n in script:
        for _ in range(n):
            t += DT
            state, events = fsm.update(pose, t)
            steps.append([pose, t])
            out.append([state, [[e.kind, e.arg] for e in events], fsm.cursor_frozen])
    return {"steps": steps, "out": out}


def _dwell(rng):
    d, t, steps, out = DwellDetector(), 0.0, [], []
    spots = [(100.0, 100.0)] * 40 + [(104.0, 103.0)] * 10 + [(300.0, 200.0)] * 45 + [None] * 3 + [(310.0, 205.0)] * 40
    for i, xy in enumerate(spots):
        t += DT
        active = not 60 <= i < 70
        xy = None if xy is None else (xy[0] + float(rng.normal(0, 1.0)), xy[1] + float(rng.normal(0, 1.0)))
        event, progress = d.update(xy, t, active)
        steps.append([None if xy is None else list(xy), t, active])
        out.append([None if event is None else event.kind, progress])
    return {"steps": steps, "out": out}


def _block(rng):
    """A made-up pointing-test block, shaped like the ones the test page records."""
    sequences = []
    for d, w in ((320.0, 96.0), (480.0, 64.0), (640.0, 48.0)):
        targets = [(700 + d / 2 * np.cos(-np.pi / 2 + 2 * np.pi * i / 9), 450 + d / 2 * np.sin(-np.pi / 2 + 2 * np.pi * i / 9)) for i in range(9)]
        trials, prev, clock = [], targets[0], 1000.0
        for k in range(1, 10):
            src, dst = targets[((k - 1) * 5) % 9], targets[(k * 5) % 9]
            click = (dst[0] + float(rng.normal(0, w / 5)), dst[1] + float(rng.normal(0, w / 5)))
            took = float(rng.uniform(600, 1400))
            path = [[clock + took * f, prev[0] + (click[0] - prev[0]) * f + float(rng.normal(0, 6)),
                     prev[1] + (click[1] - prev[1]) * f + float(rng.normal(0, 6))] for f in (0.2, 0.4, 0.6, 0.8, 0.95)]
            trials.append({"src": list(src), "dst": list(dst), "start": list(prev), "click": list(click), "t0": clock, "t1": clock + took,
                           "hit": bool(np.hypot(click[0] - dst[0], click[1] - dst[1]) <= w / 2), "path": path})
            prev, clock = click, clock + took
        sequences.append({"D": d, "W": w, "n": 9, "trials": trials})
    block = {"kind": "test", "sequences": sequences}
    flat = {"D": 100.0, "W": 20.0, "n": 1, "trials": [dict(sequences[0]["trials"][0], path=[])]}      # one click: no spread
    return {"block": block, "summary": block_summary(block), "sequences": [sequence_stats(s) for s in sequences],
            "flat": {"seq": flat, "stats": sequence_stats(flat)}}


def _session(model, hands):
    """One short session through the real Session and live loop: palm, click, drag, right click, scroll,
    hand away. Records, per frame, the pose, the state and every call the mouse received."""
    clf = Exported(model)
    base = {}
    for pose in ("palm", "pinch", "v_sign", "thumb_down"):
        best = max(hands[pose], key=lambda pts: float(clf.proba(pts).max()))
        base[pose] = best - np.array([best[F.MIDDLE_MCP, 0] - 0.5, best[F.MIDDLE_MCP, 1] - 0.5, 0.0])
    script = [("palm", 15), ("pinch", 6), ("palm", 12), ("pinch", 24), ("palm", 12), ("v_sign", 18), ("palm", 9),
              ("thumb_down", 15), ("palm", 9), (None, 9), ("palm", 9)]
    mouse = FakeMouse()
    mouse.size = lambda: SCREEN
    runner = ActionRunner(mouse, SCREEN)
    runner.toggle()
    blank = np.full((VIEW[1], VIEW[0], 3), 120, np.uint8)
    frames, out, i = [], [], 0
    with tempfile.TemporaryDirectory() as root:
        s = Session(runner, RuleClassifier(), clf, root=root, sounds=Sounds(enabled=False, player=lambda p: None),
                    effects=Effects(run=lambda cmd: None), talker=Talker(),
                    helper=Helper(lambda *a: None, lambda: None, SCREEN))
        s.pointer.mode, s.hints_on = "smooth", False
        for pose, n in script:
            for _ in range(n):
                t = (i + 1) * DT
                pts = None if pose is None else base[pose] + np.array([0.10 * np.sin(i / 23.0), 0.06 * np.cos(i / 31.0), 0.0])
                before = len(mouse.calls)
                s.tick(blank, pts, "Right" if pose else None, t)
                frames.append({"t": t, "pts": None if pts is None else pts.tolist()})
                out.append({"label": s.mode.label, "state": s.fsm.state, "calls": [list(c) for c in mouse.calls[before:]]})
                i += 1
        s.close()
    return {"view": list(VIEW), "screen": list(SCREEN), "frames": frames, "out": out}


def build(root, seed=0):
    """-> (model dict, fixture dict)."""
    clf = PoseClassifier.load(os.path.join(root, "models", "pose_mlp.joblib"), os.path.join(root, "models", "labels.json"))
    model = export_model(clf.model, clf.labels)
    data = os.path.join(root, "data")
    hands = {}
    for name in sorted(os.listdir(data)):
        if name.endswith(".csv") and not name.endswith(("_calib.csv", "_signs.csv")):
            for pose, rows in read_hands(os.path.join(data, name)).items():
                hands.setdefault(pose, []).extend(rows)
    rng = np.random.default_rng(seed)
    runner = ActionRunner(FakeMouse(), SCREEN)
    points = [(0.5, 0.5), (0.15, 0.15), (0.85, 0.80), (0.0, 1.0), (0.3, 0.7)]
    fixture = {
        "constants": {"fc_min": FC_MIN, "fc_min_preclick": FC_MIN_PRECLICK, "beta": BETA, "preclick_ratio": PRECLICK_RATIO,
                      "region": list(ACTIVE_REGION)},
        "hands": _hands(model, hands), "vote": _vote(rng), "filter": _filter(rng), "pointer": _pointer(rng),
        "fsm": _fsm(), "dwell": _dwell(rng), "stats": _block(rng),
        "to_screen": [{"n": list(p), "out": list(runner.to_screen(*p))} for p in points],
        "session": _session(model, hands),
    }
    return model, fixture


def main(root=None, out_dir=None):
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = out_dir or os.path.join(root, "try")
    model, fixture = build(root)
    os.makedirs(out_dir, exist_ok=True)
    for name, obj in (("model.json", model), ("parity.json", fixture)):
        with open(os.path.join(out_dir, name), "w") as f:
            json.dump(obj, f, separators=(",", ":"))
        print(f"wrote {os.path.join(out_dir, name)} ({os.path.getsize(os.path.join(out_dir, name)) / 1024:.0f} KB)")
    kinds = [c[0] for row in fixture["session"]["out"] for c in row["calls"] if c[0] != "move"]
    print("session script produced:", kinds)
    return model, fixture


if __name__ == "__main__":
    main()
