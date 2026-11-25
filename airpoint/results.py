"""venv/bin/python -m airpoint.results     rebuild results.js, the dashboard's only source of numbers.

Every figure here is either measured on this project's own data or produced by a simulation that is
named as one. A section with nothing behind it is None / empty, and the dashboard says so.
"""
import json
import os
import re
import shutil
import subprocess
import time

import numpy as np

from . import features as F
from .classifier import PoseClassifier, RuleClassifier
from .features import normalize_63
from .poses import POSES
from .lab import store
from .lab.stats import aggregate
from .pointer import BETA, FC_MIN, PointerPipeline
from .sim_pointer import dwell_rate, pointer_session, shake_left
from .train import load_rows

TREMOR_LEVELS = (("moderate", 6.0, 14.0), ("severe", 5.0, 25.0), ("very severe", 6.0, 40.0))   # name, Hz, peak px


def _dataset(data_dir):
    """-> (dataset section, X, y), or (None, None, None) when nothing is recorded."""
    try:
        X, y, users = load_rows(data_dir)
    except FileNotFoundError:
        return None, None, None
    poses = sorted(set(y))
    return {"rows": int(len(y)), "users": sorted(set(users)),
            "per_pose": {p: int(np.sum(y == p)) for p in poses}}, X, y


def _classifier(models_dir, dataset, X, y):
    """The trained model's own report, named for what it is, beside the rule baseline on the same frames."""
    path = os.path.join(models_dir, "train_report.json")
    if dataset is None or not os.path.exists(path):
        return None
    with open(path) as f:
        report = json.load(f)
    rules = RuleClassifier()
    right = np.array([rules.predict(row)[0] == label for row, label in zip(X, y)])
    many = len(report["users"]) >= 2
    per_pose = []
    for pose in sorted(report["per_pose"]):
        p, r = report["per_pose"][pose]["precision"], report["per_pose"][pose]["recall"]
        per_pose.append({"pose": pose, "rows": dataset["per_pose"].get(pose, 0),
                         "rule_acc": float(right[y == pose].mean()) if np.any(y == pose) else None,
                         "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0})
    return {"method": (f"leave-one-user-out over {len(report['users'])} users" if many
                       else "25% row hold-out of one user (not a generalization number)"),
            "generalizes": many, "accuracy": report["louo_acc"], "rule_acc": float(right.mean()),
            "mirrored_acc": report.get("mirrored_acc"),
            "users": report["users"], "trained_rows": report["n_rows"], "trained_at": report["trained_at"],
            "stale": report["n_rows"] != dataset["rows"], "per_pose": per_pose}


def _latency(models_dir, X):
    """Classifier time per frame, measured on real recorded rows."""
    model, labels = os.path.join(models_dir, "pose_mlp.joblib"), os.path.join(models_dir, "labels.json")
    if X is None or not os.path.exists(model):
        return None
    clf, rows = PoseClassifier.load(model, labels), X[:: max(len(X) // 300, 1)][:300]
    clf.proba(rows[0])
    features, total = [], []
    for row in rows:
        t0 = time.perf_counter()
        normalize_63(row)
        t1 = time.perf_counter()
        clf.proba(row)
        features.append(t1 - t0)
        total.append(time.perf_counter() - t1)
    return {"classifier_ms": float(np.median(total) * 1000), "features_ms": float(np.median(features) * 1000),
            "frames": len(rows)}


def _signs(models_dir, X, y):
    """One real recorded hand per pose (the middle frame of its recording): its shape, the ratios the
    hand-written rules measure on it, and what the rules and the trained network each call it."""
    if X is None:
        return None
    model, labels = os.path.join(models_dir, "pose_mlp.joblib"), os.path.join(models_dir, "labels.json")
    clf, rules, out = (PoseClassifier.load(model, labels) if os.path.exists(model) else None), RuleClassifier(), []
    tips = (("thumb", F.THUMB_TIP), ("index", F.INDEX_TIP), ("middle", F.MIDDLE_TIP), ("ring", F.RING_TIP), ("pinky", F.PINKY_TIP))
    for pose in POSES:
        rows = X[y == pose]
        if not len(rows):
            continue
        pts = F.to_array(rows[len(rows) // 2])
        palm = np.linalg.norm(pts[F.MIDDLE_MCP, :2] - pts[F.WRIST, :2]) + 1e-6
        said, conf = clf.predict(pts) if clf else (None, None)
        out.append({"pose": pose, "pts": pts[:, :2].tolist(), "pinch": F.pinch_ratio(pts),
                    "ext": {name: float(np.linalg.norm(pts[i, :2] - pts[F.WRIST, :2]) / palm) for name, i in tips},
                    "rule": rules.predict(pts)[0], "mlp": said, "conf": conf})
    return out


def _helper(root):
    """The AI helper's measurements, as written by `python -m airpoint.helper_eval`. None until that has run."""
    path = os.path.join(root, "results", "helper_eval.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        r = json.load(f)
    p, errands = r.get("pointing"), r.get("errands") or []
    return {"model": r.get("model"), "date": r.get("date"), "what": r.get("what"),
            "seconds_per_step": r.get("median_seconds_per_step"), "calls": r.get("calls"),
            "pointing": None if not p else {k: p[k] for k in ("n", "hits", "median_err_px", "max_err_px", "smallest_box")},
            "errands": [{"name": e["name"], "passed": e["passed"], "steps": len(e["steps"]),
                         "asked": e["asked_yes_or_no"], "answered": e["answered"]} for e in errands]}


def _web(root):
    """How many of the browser port's answers agree with Python's, counted now by running the check."""
    script, node = os.path.join(root, "try", "parity.mjs"), shutil.which("node")
    if not node or not os.path.exists(script):
        return None
    try:
        out = subprocess.run([node, script], capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    found = re.search(r"(\d+) of (\d+) checks agree", out)
    return {"agree": int(found.group(1)), "checks": int(found.group(2))} if found else None


def _run(session, mode):
    p = PointerPipeline()
    p.mode = mode
    return [p.update((float(x), float(y)), float(t)) for t, (x, y) in zip(session.t, session.raw)]


def _tremor(seconds, seeds):
    """Smoothing only vs canceller + smoothing on synthetic sessions. A simulation, and labelled as one."""
    levels = []
    for name, hz, amp in TREMOR_LEVELS:
        acc = {"smooth": [], "full": []}
        for seed in seeds:
            s = pointer_session(seconds, seed=seed, hz=hz, amp=amp)
            for mode in acc:
                out = _run(s, mode)
                acc[mode].append((shake_left(s, out), dwell_rate(s, out)))
        row = {"name": name, "hz": hz, "amp_px": amp}
        for mode, vals in acc.items():
            row[mode] = {"shake_px": float(np.mean([v[0] for v in vals])), "dwell_ok": float(np.mean([v[1] for v in vals]))}
        levels.append(row)
    return {"note": "Simulation: synthetic reaches and holds with a tremor of wandering frequency and size, 30 fps, "
                    "640-wide frame. No patient data.",
            "smooth": "1-Euro smoothing only", "full": "tremor canceller + 1-Euro smoothing",
            "fc_min": FC_MIN, "beta": BETA, "sessions": len(seeds), "seconds": seconds, "levels": levels}


def _pointing(path):
    blocks = [b for b in store.read(path) if b.get("kind") == "test"]
    return {"conditions": aggregate(blocks), "participants": len({b.get("participant") for b in blocks}),
            "blocks": len(blocks)}


def _commit(root):
    try:
        return subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _rounded(obj, digits=4):
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: _rounded(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_rounded(v, digits) for v in obj]
    return obj


def build(root, tremor_seconds=60, tremor_seeds=(0, 1, 2)):
    data_dir, models_dir = os.path.join(root, "data"), os.path.join(root, "models")
    dataset, X, y = _dataset(data_dir)
    return _rounded({
        "generated_at": time.strftime("%Y-%m-%d %H:%M"), "commit": _commit(root),
        "dataset": dataset, "classifier": _classifier(models_dir, dataset, X, y), "latency": _latency(models_dir, X),
        "tremor": _tremor(tremor_seconds, tremor_seeds),
        "pointing": _pointing(os.path.join(root, "results", "trials.jsonl")),
        "signs": _signs(models_dir, X, y), "helper": _helper(root), "web": _web(root),
    })


def write(root, **kw):
    path = os.path.join(root, "results.js")
    with open(path, "w") as f:
        f.write("window.AIRPOINT_RESULTS = " + json.dumps(build(root, **kw), indent=2) + ";\n")
    return path


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = write(root)
    with open(path) as f:
        r = json.loads(f.read()[len("window.AIRPOINT_RESULTS = "):].rstrip().rstrip(";"))
    d, c, p = r["dataset"], r["classifier"], r["pointing"]
    print(f"dataset   : {d['rows']} rows, users {', '.join(d['users'])}" if d else "dataset   : nothing recorded")
    print(f"classifier: {c['accuracy'] * 100:.1f}% by {c['method']}; rules {c['rule_acc'] * 100:.1f}% on the same frames"
          if c else "classifier: no training report (run python -m airpoint.train)")
    for lvl in r["tremor"]["levels"]:
        print(f"tremor sim: {lvl['name']:<12} shake left {lvl['smooth']['shake_px']:.1f} -> {lvl['full']['shake_px']:.1f} px, "
              f"dwell works {lvl['smooth']['dwell_ok'] * 100:.0f}% -> {lvl['full']['dwell_ok'] * 100:.0f}%")
    print(f"pointing  : {p['blocks']} blocks from {p['participants']} people" if p["blocks"]
          else "pointing  : not measured yet (run python -m airpoint.lab)")
    h, w = r["helper"], r["web"]
    print(f"AI helper : {h['pointing']['hits']}/{h['pointing']['n']} pointing, "
          f"{sum(e['passed'] for e in h['errands'])}/{len(h['errands'])} errands ({h['model']}, {h['date']}, drawn screens)"
          if h and h["pointing"] else "AI helper : not measured yet (run python -m airpoint.helper_eval)")
    print(f"browser   : {w['agree']} of {w['checks']} parity checks agree" if w else "browser   : parity not checked (needs node)")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
