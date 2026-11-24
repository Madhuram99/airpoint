import json

import numpy as np

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

    report = train_personal(str(data), str(data / "b_calib.csv"), str(models), "b", seed=0)

    assert report["n_calib"] == 180 and report["acc_on_own_frames"] >= 0.9
    assert report["model"] == str(models / "personal_b.joblib")
    assert (models / "personal_b.joblib").exists() and (models / "personal_b_labels.json").exists()


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
