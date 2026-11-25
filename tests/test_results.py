import json

import numpy as np

from airpoint import results
from airpoint.lab import store
from airpoint.train import train_and_save
from tests.test_lab_stats import block
from tests.test_train import _write_user

QUICK = dict(tremor_seconds=30, tremor_seeds=(0,))


def project(tmp_path, users=("a", "b")):
    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(0)
    for i, u in enumerate(users):
        _write_user(data / f"{u}.csv", u, rng, 1.0 + 0.2 * i, np.array([0.05 * i, 0.0, 0.0]))
    train_and_save(str(data), str(tmp_path / "models"), seed=0)
    return str(tmp_path)


def test_build_reports_only_what_the_project_actually_has(tmp_path):
    root = project(tmp_path)
    for b in (block("P1", "mouse"), block("P2", "mouse"), block("P1", "airpoint sign-click")):
        store.append(str(tmp_path / "results" / "trials.jsonl"), b)

    r = results.build(root, **QUICK)

    assert r["dataset"]["rows"] == 360 and r["dataset"]["users"] == ["a", "b"]
    c = r["classifier"]
    assert c["generalizes"] and c["method"] == "leave-one-user-out over 2 users"
    assert [p["pose"] for p in c["per_pose"]] == sorted(p["pose"] for p in c["per_pose"]) and len(c["per_pose"]) == 6
    assert all(p["rows"] == 60 and p["rule_acc"] >= 0.95 and 0.0 <= p["f1"] <= 1.0 for p in c["per_pose"])
    assert 0.9 <= c["accuracy"] <= 1.0 and c["rule_acc"] >= 0.95 and not c["stale"]
    assert r["latency"]["classifier_ms"] > 0
    assert [row["condition"] for row in r["pointing"]["conditions"]] == ["airpoint sign-click", "mouse"]
    assert r["pointing"]["participants"] == 2 and r["pointing"]["blocks"] == 3


def test_one_user_is_reported_as_a_hold_out_not_as_generalization(tmp_path):
    c = results.build(project(tmp_path, users=("solo",)), **QUICK)["classifier"]
    assert not c["generalizes"]
    assert c["method"] == "25% row hold-out of one user (not a generalization number)"


def test_tremor_table_is_labelled_simulation_and_shows_the_canceller_helping(tmp_path):
    t = results.build(str(tmp_path), **QUICK)["tremor"]
    assert "simulation" in t["note"].lower()
    assert [lvl["name"] for lvl in t["levels"]] == ["moderate", "severe", "very severe"]
    severe = t["levels"][1]
    assert severe["full"]["shake_px"] < 0.5 * severe["smooth"]["shake_px"]
    assert severe["full"]["dwell_ok"] > severe["smooth"]["dwell_ok"]


def test_an_empty_project_says_nothing_was_measured_instead_of_inventing_numbers(tmp_path):
    r = results.build(str(tmp_path), **QUICK)
    assert r["dataset"] is None and r["classifier"] is None and r["latency"] is None
    assert r["pointing"] == {"conditions": [], "participants": 0, "blocks": 0}
    path = results.write(str(tmp_path), **QUICK)
    text = open(path).read()
    assert text.startswith("window.AIRPOINT_RESULTS = ") and text.rstrip().endswith(";")
    assert json.loads(text[len("window.AIRPOINT_RESULTS = "):].rstrip().rstrip(";"))["dataset"] is None


def test_signs_are_real_recorded_hands_with_what_each_classifier_says(tmp_path):
    r = results.build(project(tmp_path), **QUICK)
    assert [s["pose"] for s in r["signs"]] == ["palm", "pinch", "fist", "v_sign", "thumb_up", "thumb_down"]
    for s in r["signs"]:
        assert len(s["pts"]) == 21 and len(s["pts"][0]) == 2 and s["mlp"] == s["pose"] and 0.5 < s["conf"] <= 1.0
        assert set(s["ext"]) == {"thumb", "index", "middle", "ring", "pinky"} and s["pinch"] > 0 and s["rule"] in results.POSES


def test_helper_and_browser_sections_are_empty_until_measured(tmp_path):
    root = project(tmp_path)
    r = results.build(root, **QUICK)
    assert r["helper"] is None and r["web"] is None
    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "results" / "helper_eval.json").write_text(json.dumps({
        "model": "gpt-6.1-sol", "date": "2026-11-25", "what": "drawn screens", "calls": 9, "median_seconds_per_step": 3.5,
        "pointing": {"n": 4, "hits": 4, "median_err_px": 1.0, "max_err_px": 1.6, "median_seconds": 3.5, "smallest_box": [52, 27], "rows": []},
        "errands": [{"name": "reply and send", "request": "x", "passed": True, "steps": ["a", "b"], "asked_yes_or_no": ['click "Send"'],
                     "answered": "yes", "ended": "Sent.", "seconds": 20.0}]}))
    h = results.build(root, **QUICK)["helper"]
    assert h["pointing"] == {"n": 4, "hits": 4, "median_err_px": 1.0, "max_err_px": 1.6, "smallest_box": [52, 27]}
    assert h["errands"] == [{"name": "reply and send", "passed": True, "steps": 2, "asked": ['click "Send"'], "answered": "yes"}]
    assert h["seconds_per_step"] == 3.5 and h["model"] == "gpt-6.1-sol"


def test_the_real_project_counts_its_browser_parity_checks():
    import os
    import shutil
    import pytest
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    w = results._web(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert w["agree"] == w["checks"] > 3000
