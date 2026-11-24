import numpy as np
from airpoint.classifier import PoseClassifier, RuleClassifier, VoteSmoother
from airpoint.poses import NONE, UNKNOWN
from tests.hands import BY_POSE


def test_rules_classify_every_synthetic_pose():
    clf = RuleClassifier()
    for pose, pts in BY_POSE.items():
        label, conf = clf.predict(pts)
        assert label == pose, f"{pose} read as {label}"
        assert conf == 1.0


def test_rules_report_name():
    assert RuleClassifier().name == "rules"


class _Model:
    classes_ = np.array(["fist", "palm", "pinch"])

    def predict_proba(self, x):
        assert x.shape == (1, 63)
        return np.array([[0.1, 0.2, 0.7]])


def test_pose_classifier_returns_label_from_labels_list_and_confidence():
    clf = PoseClassifier(_Model(), ["fist", "palm", "pinch"])
    assert clf.predict(BY_POSE["pinch"]) == ("pinch", 0.7)
    assert clf.name == "mlp"


def test_smoother_ignores_one_or_two_frame_flicker():
    s = VoteSmoother(window=5, min_conf=0.75)
    for _ in range(5):
        s.update("palm", 0.9)
    assert s.update("pinch", 0.9) == "palm"
    assert s.update("pinch", 0.9) == "palm"


def test_smoother_switches_on_majority():
    s = VoteSmoother(window=5, min_conf=0.75)
    for _ in range(5):
        s.update("palm", 0.9)
    s.update("pinch", 0.9)
    s.update("pinch", 0.9)
    assert s.update("pinch", 0.9) == "pinch"


def test_smoother_low_confidence_is_unknown():
    s = VoteSmoother(window=5, min_conf=0.75)
    for _ in range(4):
        s.update("pinch", 0.5)
    assert s.update("pinch", 0.5) == UNKNOWN


def test_smoother_no_hand_is_none_and_clears_history():
    s = VoteSmoother(window=5, min_conf=0.75)
    for _ in range(5):
        s.update("palm", 0.9)
    assert s.update(None, 0.0) == NONE
    assert s.update("pinch", 0.9) == "pinch"


def test_proba_is_in_label_order_and_agrees_with_predict():
    clf = PoseClassifier(_Model(), ["fist", "palm", "pinch"])
    p = clf.proba(BY_POSE["pinch"])
    assert p.shape == (3,) and abs(p.sum() - 1.0) < 1e-9
    assert clf.labels[int(np.argmax(p))] == clf.predict(BY_POSE["pinch"])[0]


def test_rules_proba_is_one_hot_over_the_six_poses():
    clf = RuleClassifier()
    assert clf.labels == ["palm", "pinch", "fist", "v_sign", "thumb_up", "thumb_down"]
    for pose, pts in BY_POSE.items():
        p = clf.proba(pts)
        assert p.sum() == 1.0 and clf.labels[int(np.argmax(p))] == pose
