"""Pointing-test statistics, following ISO 9241-411 as recommended by Soukoreff & MacKenzie (2004).

A sequence is one difficulty level: the same D and W for every selection. Along the task axis
(previous target centre -> this target centre):
    dx = where the click landed relative to the target centre
    We = 4.133 x SD(dx)        effective width: the target the user actually behaved as if aiming at
    Ae = mean movement length  effective distance: from the previous click to this one
    IDe = log2(Ae / We + 1),  throughput TP = IDe / mean movement time     (bits per second)
"""
import math
from collections import defaultdict

import numpy as np


def _trial_path(tr, w):
    """-> (path length / straight distance, re-entries into the target) for one selection."""
    start, click, dst = np.array(tr["start"], float), np.array(tr["click"], float), np.array(tr["dst"], float)
    pts = np.array([start] + [p[1:3] for p in tr.get("path", [])] + [click], float)
    straight = float(np.linalg.norm(click - start))
    length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    inside = np.linalg.norm(pts - dst, axis=1) <= w / 2.0
    entries = int(inside[0]) + int(np.sum(~inside[:-1] & inside[1:]))
    return (length / straight if straight > 0 else 1.0), max(entries - 1, 0)


def sequence_stats(seq):
    """One difficulty level -> its statistics. `flat` = the clicks had no spread along the axis,
    so the nominal width was used instead of the effective one."""
    dx, ae, mt, per_trial = [], [], [], []
    for tr in seq["trials"]:
        src, dst = np.array(tr["src"], float), np.array(tr["dst"], float)
        u = (dst - src) / np.linalg.norm(dst - src)
        click, start = np.array(tr["click"], float), np.array(tr["start"], float)
        dx.append(float((click - dst) @ u))
        ae.append(float((click - start) @ u))
        mt.append((tr["t1"] - tr["t0"]) / 1000.0)
        ratio, reentries = _trial_path(tr, seq["W"])
        per_trial.append({"mt": mt[-1], "dx": dx[-1], "hit": bool(tr["hit"]), "path_ratio": ratio, "reentries": reentries})
    n = len(dx)
    sdx = float(np.std(dx, ddof=1)) if n >= 2 else 0.0
    flat = sdx < 1e-6
    we = float(seq["W"]) if flat else 4.133 * sdx
    ae_mean, mt_mean = float(np.mean(ae)), float(np.mean(mt))
    ide = math.log2(ae_mean / we + 1.0) if ae_mean > 0 else 0.0
    return {"D": seq["D"], "W": seq["W"], "n": n, "errors": sum(not t["hit"] for t in per_trial),
            "sdx": sdx, "we": we, "ae": ae_mean, "ide": ide, "mt": mt_mean,
            "tp": ide / mt_mean if mt_mean > 0 else 0.0, "flat": flat,
            "path_ratio": float(np.mean([t["path_ratio"] for t in per_trial])),
            "reentries": float(np.mean([t["reentries"] for t in per_trial])), "trials": per_trial}


def block_summary(block):
    """One saved block -> what the page shows and what the dashboard aggregates."""
    if block.get("kind") == "tutorial":
        steps = block.get("steps", [])
        return {"kind": "tutorial", "seconds": float(sum(s["seconds"] for s in steps)), "steps": len(steps)}
    seqs = [sequence_stats(s) for s in block["sequences"] if s["trials"]]
    trials = sum(s["n"] for s in seqs)
    return {"kind": "test", "tp": float(np.mean([s["tp"] for s in seqs])), "trials": trials,
            "error_rate": sum(s["errors"] for s in seqs) / trials,
            "mt": sum(s["mt"] * s["n"] for s in seqs) / trials,
            "path_ratio": float(np.mean([s["path_ratio"] for s in seqs])),
            "reentries": float(np.mean([s["reentries"] for s in seqs])),
            "flat": any(s["flat"] for s in seqs),
            "sequences": [{k: v for k, v in s.items() if k != "trials"} for s in seqs]}


def aggregate(blocks):
    """Test blocks -> one row per condition. Each participant counts once (their blocks are averaged
    first), so one keen tester cannot dominate the mean."""
    by = defaultdict(lambda: defaultdict(list))
    for b in blocks:
        if b.get("kind") == "test" and b.get("sequences"):
            by[b.get("condition", "unlabelled")][b.get("participant", "?")].append(block_summary(b))
    rows = []
    for condition, people in sorted(by.items()):
        per_person = [float(np.mean([s["tp"] for s in ss])) for ss in people.values()]
        every = [s for ss in people.values() for s in ss]
        trials = sum(s["trials"] for s in every)
        rows.append({"condition": condition, "participants": len(people), "blocks": len(every), "trials": trials,
                     "tp": float(np.mean(per_person)),
                     "tp_sd": float(np.std(per_person, ddof=1)) if len(per_person) >= 2 else None,
                     "error_rate": sum(s["error_rate"] * s["trials"] for s in every) / trials,
                     "mt": sum(s["mt"] * s["trials"] for s in every) / trials,
                     "path_ratio": float(np.mean([s["path_ratio"] for s in every])),
                     "reentries": float(np.mean([s["reentries"] for s in every]))})
    return rows
