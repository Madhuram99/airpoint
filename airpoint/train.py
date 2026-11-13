"""Train the 63-input pose MLP on data/*.csv. Reports leave-one-user-out accuracy.

    venv/bin/python -m airpoint.train            # data/ -> models/
"""
import csv
import glob
import json
import os
import sys
import time

import joblib
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier

from .features import normalize_63

TILT_MAX_RAD = 0.60   # +/- 34 degrees in-plane
NOISE_SIGMA = 0.12    # palm units
AUG_COPIES = 3


def load_rows(data_dir):
    X, y, users = [], [], []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                X.append([float(row[f"{a}{i}"]) for i in range(21) for a in "xyz"])
                y.append(row["pose"])
                users.append(row["user"])
    if not X:
        raise SystemExit(f"no rows in {data_dir}/*.csv — record first: run_demo.py --record --user <name>")
    return np.array(X), np.array(y), np.array(users)


def normalize_rows(X_raw):
    return np.array([normalize_63(r) for r in X_raw])


def augment(Xn, y, copies, rng):
    """Rotate each normalized hand in-plane, add landmark noise, re-normalize."""
    out_X, out_y = [Xn], [y]
    for _ in range(copies):
        pts = Xn.reshape(-1, 21, 3).copy()
        theta = rng.uniform(-TILT_MAX_RAD, TILT_MAX_RAD, size=len(pts))
        c, s = np.cos(theta)[:, None], np.sin(theta)[:, None]
        x, yy = pts[:, :, 0].copy(), pts[:, :, 1].copy()
        pts[:, :, 0], pts[:, :, 1] = c * x - s * yy, s * x + c * yy
        pts += rng.normal(0.0, NOISE_SIGMA, size=pts.shape)
        out_X.append(np.array([normalize_63(p) for p in pts]))
        out_y.append(y)
    return np.concatenate(out_X), np.concatenate(out_y)


def make_model(seed):
    return MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
                         max_iter=400, random_state=seed)


def train_and_save(data_dir, models_dir, seed=42):
    rng = np.random.default_rng(seed)
    X_raw, y, users = load_rows(data_dir)
    Xn = normalize_rows(X_raw)
    labels = sorted(set(y))

    louo, y_true, y_pred = {}, [], []
    for u in sorted(set(users)):
        tr, te = users != u, users == u
        X_tr, y_tr = augment(Xn[tr], y[tr], AUG_COPIES, rng)
        m = make_model(seed).fit(X_tr, y_tr)
        p = m.predict(Xn[te])
        louo[u] = float(np.mean(p == y[te]))
        y_true.extend(y[te]); y_pred.extend(p)
    prec, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    t0 = time.perf_counter()
    X_all, y_all = augment(Xn, y, AUG_COPIES, rng)
    model = make_model(seed).fit(X_all, y_all)
    fit_s = time.perf_counter() - t0

    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, "pose_mlp.joblib"))
    with open(os.path.join(models_dir, "labels.json"), "w") as f:
        json.dump(list(model.classes_), f)
    report = {
        "n_rows": int(len(y)), "users": sorted(set(users)), "labels": labels,
        "louo": louo, "louo_acc": float(np.mean(list(louo.values()))),
        "per_pose": {l: {"precision": float(p), "recall": float(r)} for l, p, r in zip(labels, prec, rec)},
        "fit_seconds": fit_s, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(models_dir, "train_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def main(argv=None):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = train_and_save(os.path.join(root, "data"), os.path.join(root, "models"))
    print(f"rows {r['n_rows']} | users {', '.join(r['users'])} | fit {r['fit_seconds']:.1f}s")
    print(f"leave-one-user-out accuracy: {r['louo_acc'] * 100:.1f}%  "
          + "  ".join(f"{u} {a * 100:.0f}%" for u, a in r["louo"].items()))
    print(f"{'pose':<12}{'precision':>10}{'recall':>8}")
    for l, m in r["per_pose"].items():
        print(f"{l:<12}{m['precision']:>10.3f}{m['recall']:>8.3f}")
    print("saved models/pose_mlp.joblib, models/labels.json, models/train_report.json")


if __name__ == "__main__":
    main(sys.argv[1:])
