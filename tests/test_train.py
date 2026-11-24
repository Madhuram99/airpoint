import json

import numpy as np
import pytest

from airpoint.recorder import Recorder
from airpoint.train import train_and_save, train_personal
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


def test_personal_model_scores_calibration_holdout(tmp_path):
    rng = np.random.default_rng(2)
    data, models = tmp_path / "data", tmp_path / "models"
    data.mkdir()
    _write_user(data / "a.csv", "a", rng, 1.0, np.array([0.0, 0.0, 0.0]))
    _write_user(data / "b_calib.csv", "b", rng, 1.3, np.array([0.1, -0.05, 0.0]))

    report = train_personal(str(data), str(models), "b", seed=0)

    assert report["n_personal"] == 180 and report["acc_on_own_frames"] >= 0.9
    assert report["custom"] == [] and report["false_reads"] == {}
    assert set(report["per_label"]) == set(BY_POSE) and all(v["recall"] >= 0.8 for v in report["per_label"].values())
    assert report["model"] == str(models / "personal_b.joblib")
    assert (models / "personal_b.joblib").exists() and (models / "personal_b_labels.json").exists()


def _write_sign(path, user, name, pts, rng, n=80):
    r = Recorder(str(path), user)
    for _ in range(n):
        r.write(name, 0.0, "Right", pts + rng.normal(0.0, 0.006, size=pts.shape))
    r.close()


def test_a_taught_sign_becomes_a_class_with_its_own_numbers(tmp_path):
    from tests.hands import ROCK
    rng = np.random.default_rng(5)
    data, models = tmp_path / "data", tmp_path / "models"
    data.mkdir()
    _write_user(data / "a.csv", "a", rng, 1.0, np.array([0.0, 0.0, 0.0]))
    _write_sign(data / "b_signs.csv", "b", "rock", ROCK, rng)

    report = train_personal(str(data), str(models), "b", seed=0)

    assert report["custom"] == ["rock"] and report["n_personal"] == 80
    assert report["per_label"]["rock"]["recall"] >= 0.9
    assert report["false_reads"]["rock"] <= 0.02           # the shared recordings are not read as "rock"
    assert "rock" in json.loads((models / "personal_b_labels.json").read_text())
    assert train_and_save(str(data), str(tmp_path / "shared"), seed=0)["n_rows"] == 180   # signs stay out of the shared model


def test_personal_training_needs_enough_frames(tmp_path):
    rng = np.random.default_rng(6)
    data = tmp_path / "data"
    data.mkdir()
    _write_user(data / "a.csv", "a", rng, 1.0, np.array([0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="only 0 usable frames"):
        train_personal(str(data), str(tmp_path / "models"), "nobody", seed=0)


def test_base_training_ignores_calibration_files(tmp_path):
    rng = np.random.default_rng(3)
    data, models = tmp_path / "data", tmp_path / "models"
    data.mkdir()
    _write_user(data / "a.csv", "a", rng, 1.0, np.array([0.0, 0.0, 0.0]))
    _write_user(data / "b_calib.csv", "b", rng, 1.3, np.array([0.1, -0.05, 0.0]))

    report = train_and_save(str(data), str(models), seed=0)

    assert report["users"] == ["a"] and report["n_rows"] == 180


def test_training_makes_one_model_read_either_hand(tmp_path):
    from airpoint.classifier import PoseClassifier
    rng = np.random.default_rng(4)
    data, models = tmp_path / "data", tmp_path / "models"
    data.mkdir()
    _write_user(data / "a.csv", "a", rng, 1.0, np.array([0.0, 0.0, 0.0]))
    _write_user(data / "b.csv", "b", rng, 1.25, np.array([0.1, -0.05, 0.0]))

    report = train_and_save(str(data), str(models), seed=0)

    assert report["mirrored_acc"] >= 0.9
    clf = PoseClassifier.load(str(models / "pose_mlp.joblib"), str(models / "labels.json"))
    for pose, pts in BY_POSE.items():
        left = pts.copy()
        left[:, 0] = 1.0 - left[:, 0]                 # the same sign made by the other hand
        assert clf.predict(left)[0] == pose, pose
