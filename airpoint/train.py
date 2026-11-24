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
from .recorder import looks_like

TILT_MAX_RAD = 0.60   # +/- 34 degrees in-plane
NOISE_SIGMA = 0.12    # palm units
AUG_COPIES = 3
PERSONAL_WEIGHT = 3        # calibration rows duplicated this many times
CALIB_SUFFIX = "_calib.csv"


def read_csv_rows(path):
    """One CSV -> (X, y, users, dropped). Rows whose geometry contradicts their label are skipped."""
    X, y, users, dropped = [], [], [], 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            coords = [float(row[f"{a}{i}"]) for i in range(21) for a in "xyz"]
            if not looks_like(row["pose"], coords):
                dropped += 1
                continue
            X.append(coords)
            y.append(row["pose"])
            users.append(row["user"])
    return X, y, users, dropped


def load_rows(data_dir):
    """Base dataset: data/*.csv minus *_calib.csv (one person's calibration, used by train_personal)."""
    X, y, users, dropped = [], [], [], 0
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        if path.endswith(CALIB_SUFFIX):
            continue
        x_, y_, u_, d_ = read_csv_rows(path)
        X += x_; y += y_; users += u_; dropped += d_
    if not X:
        raise FileNotFoundError(f"no rows in {data_dir}/*.csv — record first: run_demo.py --record --user <name>")
    if dropped:
        print(f"[train] skipped {dropped} rows whose geometry contradicts their label")
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


def folds(users, rng):
    """Leave-one-user-out. With a single user there is no unseen hand: 25% row hold-out instead."""
    names = sorted(set(users))
    if len(names) >= 2:
        for u in names:
            yield u, users != u, users == u
        return
    te = rng.random(len(users)) < 0.25
    yield f"{names[0]} (row hold-out)", ~te, te


def train_and_save(data_dir, models_dir, seed=42):
    rng = np.random.default_rng(seed)
    X_raw, y, users = load_rows(data_dir)
    Xn = normalize_rows(X_raw)
    labels = sorted(set(y))

    louo, y_true, y_pred = {}, [], []
    for name, tr, te in folds(users, rng):
        X_tr, y_tr = augment(Xn[tr], y[tr], AUG_COPIES, rng)
        m = make_model(seed).fit(X_tr, y_tr)
        p = m.predict(Xn[te])
        louo[name] = float(np.mean(p == y[te]))
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


def train_personal(data_dir, calib_path, models_dir, user, seed=42, weight=PERSONAL_WEIGHT):
    """Base rows + this user's calibration rows x`weight` -> models/personal_<user>.joblib.
    Accuracy is on a 25% hold-out of the calibration rows, which the fit never sees."""
    rng = np.random.default_rng(seed)
    X_base, y_base, _ = load_rows(data_dir)
    Xc, yc, _, _ = read_csv_rows(calib_path)
    if len(Xc) < 60:
        raise ValueError(f"only {len(Xc)} usable calibration frames; redo calibration")
    Xc, yc = np.array(Xc), np.array(yc)
    te = rng.random(len(yc)) < 0.25
    Xn_base, Xn_c = normalize_rows(X_base), normalize_rows(Xc)
    X_tr = np.concatenate([Xn_base] + [Xn_c[~te]] * weight)
    y_tr = np.concatenate([y_base] + [yc[~te]] * weight)
    X_aug, y_aug = augment(X_tr, y_tr, AUG_COPIES, rng)

    t0 = time.perf_counter()
    model = make_model(seed).fit(X_aug, y_aug)
    fit_s = time.perf_counter() - t0
    acc = float(np.mean(model.predict(Xn_c[te]) == yc[te]))

    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"personal_{user}.joblib")
    labels_path = os.path.join(models_dir, f"personal_{user}_labels.json")
    joblib.dump(model, model_path)
    with open(labels_path, "w") as f:
        json.dump(list(model.classes_), f)
    return {"user": user, "n_calib": int(len(yc)), "acc_on_own_frames": acc,
            "fit_seconds": fit_s, "model": model_path, "labels": labels_path}


def main(argv=None):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        r = train_and_save(os.path.join(root, "data"), os.path.join(root, "models"))
    except FileNotFoundError as e:
        raise SystemExit(e)
    print(f"rows {r['n_rows']} | users {', '.join(r['users'])} | fit {r['fit_seconds']:.1f}s")
    print(f"leave-one-user-out accuracy: {r['louo_acc'] * 100:.1f}%  "
          + "  ".join(f"{u} {a * 100:.0f}%" for u, a in r["louo"].items()))
    print(f"{'pose':<12}{'precision':>10}{'recall':>8}")
    for l, m in r["per_pose"].items():
        print(f"{l:<12}{m['precision']:>10.3f}{m['recall']:>8.3f}")
    print("saved models/pose_mlp.joblib, models/labels.json, models/train_report.json")


if __name__ == "__main__":
    main(sys.argv[1:])
