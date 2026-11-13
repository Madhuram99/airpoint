import json

import numpy as np

from airpoint.recorder import Recorder
from airpoint.train import train_and_save
from tests.hands import BY_POSE


def _write_user(path, user, rng, scale, shift):
    r = Recorder(str(path), user)
    for pose, pts in BY_POSE.items():
        for _ in range(30):
            jitter = rng.normal(0.0, 0.006, size=pts.shape)
            p = (pts - pts[0]) * scale + pts[0] + shift + jitter
            r.write(pose, 0.0, "Right", p)
    r.close()


def test_train_reports_louo_accuracy_and_saves_model(tmp_path):
    rng = np.random.default_rng(0)
    data, models = tmp_path / "data", tmp_path / "models"
    data.mkdir()
    _write_user(data / "a.csv", "a", rng, 1.0, np.array([0.0, 0.0, 0.0]))
    _write_user(data / "b.csv", "b", rng, 1.25, np.array([0.1, -0.05, 0.0]))

    report = train_and_save(str(data), str(models), seed=0)

    assert report["n_rows"] == 360 and sorted(report["users"]) == ["a", "b"]
    assert report["louo_acc"] >= 0.9
    assert set(report["per_pose"]) == set(BY_POSE)
    labels = json.loads((models / "labels.json").read_text())
    assert labels == sorted(BY_POSE)
    assert (models / "pose_mlp.joblib").exists()
    assert (models / "train_report.json").exists()


def test_single_user_falls_back_to_row_holdout(tmp_path):
    rng = np.random.default_rng(1)
    data, models = tmp_path / "data", tmp_path / "models"
    data.mkdir()
    _write_user(data / "solo.csv", "solo", rng, 1.0, np.array([0.0, 0.0, 0.0]))

    report = train_and_save(str(data), str(models), seed=0)

    assert list(report["louo"]) == ["solo (row hold-out)"]
    assert report["louo_acc"] >= 0.9
    assert (models / "pose_mlp.joblib").exists()
