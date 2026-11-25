"""The browser demo's numbers come from Python, and its JavaScript port must agree with Python."""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from airpoint import export_web as X
from airpoint import features as F
from airpoint.classifier import PoseClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = shutil.which("node")


@pytest.fixture(scope="module")
def built():
    return X.build(ROOT)


def test_the_exported_network_answers_like_the_trained_one(built):
    model, fixture = built
    clf = PoseClassifier.load(os.path.join(ROOT, "models", "pose_mlp.joblib"), os.path.join(ROOT, "models", "labels.json"))
    assert model["labels"] == clf.labels and [len(layer["b"]) for layer in model["layers"]] == [128, 64, 6]
    for hand in fixture["hands"]:
        pts = np.array(hand["pts"])
        assert np.abs(X.forward(model, F.normalize_63(pts)) - clf.proba(pts)).max() < 1e-5      # float32 weights
        assert hand["label"] == clf.predict(pts)[0]
    assert len(fixture["hands"]) == 30 and {h["label"] for h in fixture["hands"]} == set(clf.labels)


def test_the_session_script_exercises_every_action(built):
    _, fixture = built
    kinds = [c[0] for row in fixture["session"]["out"] for c in row["calls"] if c[0] != "move"]
    assert kinds[:4] == ["click", "down", "up", "right_click"] and set(kinds[4:]) == {"scroll"}
    states = {row["state"] for row in fixture["session"]["out"]}
    assert {"MOVING", "PRESSING", "DRAGGING", "HOLD", "SCROLLING", "RESTING"} <= states
    assert fixture["session"]["frames"][-10]["pts"] is None or any(f["pts"] is None for f in fixture["session"]["frames"])


def test_what_is_committed_is_what_python_writes_today(built, tmp_path):
    model, fixture = built
    with open(os.path.join(ROOT, "try", "model.json")) as f:
        assert json.load(f) == model
    with open(os.path.join(ROOT, "try", "parity.json")) as f:
        committed = json.load(f)
    assert committed["session"]["out"] == fixture["session"]["out"] and committed["fsm"] == fixture["fsm"]
    assert np.allclose([h["proba"] for h in committed["hands"]], [h["proba"] for h in fixture["hands"]], atol=1e-12)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_javascript_port_agrees_with_python(built, tmp_path):
    model, fixture = built
    (tmp_path / "parity.json").write_text(json.dumps(fixture))
    (tmp_path / "model.json").write_text(json.dumps(model))
    script = os.path.join(ROOT, "try", "parity.mjs")
    fresh = subprocess.run([NODE, script, str(tmp_path / "parity.json"), str(tmp_path / "model.json")],
                           capture_output=True, text=True, timeout=120)
    assert fresh.returncode == 0, fresh.stdout + fresh.stderr
    assert "checks agree with Python" in fresh.stdout and "MISMATCH" not in fresh.stdout
    committed = subprocess.run([NODE, script], capture_output=True, text=True, timeout=120)
    assert committed.returncode == 0, committed.stdout + committed.stderr


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_check_really_fails_when_the_port_is_wrong(built, tmp_path):
    model, fixture = built
    broken = json.loads(json.dumps(fixture))
    broken["session"]["out"][20]["state"] = "DRAGGING"            # Python never said this
    broken["hands"][0]["proba"][0] += 1e-6
    (tmp_path / "parity.json").write_text(json.dumps(broken))
    out = subprocess.run([NODE, os.path.join(ROOT, "try", "parity.mjs"), str(tmp_path / "parity.json")],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 1 and out.stdout.count("MISMATCH") == 2
