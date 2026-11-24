import json

import pytest

from airpoint import profile as P
from airpoint.fsm import Event


def test_round_trip_keeps_every_field(tmp_path):
    p = P.Profile(user="ana", model="models/personal_ana.joblib", labels="models/personal_ana_labels.json",
                  acc_on_own_frames=0.97, fc_min=0.55, region=[0.3, 0.3, 0.7, 0.8], dwell=True, skipped=["thumb_down"])
    path = P.save(str(tmp_path), p)
    assert path == str(tmp_path / "profiles" / "ana.json")
    assert P.load(str(tmp_path), "ana") == p
    assert P.load(str(tmp_path), "nobody") is None


def test_a_week_2_profile_is_upgraded_on_load(tmp_path):
    (tmp_path / "profiles").mkdir()
    (tmp_path / "profiles" / "ana.json").write_text(json.dumps(
        {"user": "ana", "fc_min": 0.55, "model": "models/personal_ana.joblib",
         "labels": "models/personal_ana_labels.json", "acc_on_own_frames": 0.97}))
    p = P.load(str(tmp_path), "ana")
    assert p.version == 2 and p.fc_min == 0.55 and p.model == "models/personal_ana.joblib"
    assert p.roles == P.default_roles() and p.region == list(P.DEFAULT_REGION) and not p.dwell and p.filter == "full"


def test_roles_are_checked():
    assert P.check_roles(P.default_roles()) is None
    assert "twice" in P.check_roles({"grip": ["pinch", "fist"], "right_click": ["fist"]})
    assert "unknown role" in P.check_roles({"zoom": ["palm"]})
    assert "list of pose names" in P.check_roles({"grip": "pinch"})


def test_fsm_parts_turn_roles_into_what_the_state_machine_takes():
    grip, scroll, hold = P.fsm_parts({"grip": ["pinch"], "right_click": ["fist"], "scroll_up": ["thumb_up"], "scroll_down": []})
    assert grip == frozenset({"pinch"}) and scroll == {"thumb_up": 1}
    assert hold == {"fist": Event("right_click")}


def test_dropping_poses_removes_them_from_every_role():
    roles = P.drop_poses(P.default_roles(), ["fist", "thumb_down"])
    assert roles == {"grip": ["pinch"], "right_click": ["v_sign"], "scroll_up": ["thumb_up"], "scroll_down": []}


def test_command_line_edits_and_saves(tmp_path, capsys):
    root = str(tmp_path)
    P.main(["ana", "--role", "grip=pinch", "--role", "right_click=fist", "--off", "scroll_down",
            "--dwell", "on", "--region", "0.3,0.3,0.7,0.8", "--filter", "smooth"], root=root)
    p = P.load(root, "ana")
    assert p.roles == {"grip": ["pinch"], "right_click": ["fist"], "scroll_up": ["thumb_up"], "scroll_down": []}
    assert p.dwell and p.region == [0.3, 0.3, 0.7, 0.8] and p.filter == "smooth"
    assert '"dwell": true' in capsys.readouterr().out
    P.main(["ana", "--reset-roles", "--reset-region"], root=root)
    p = P.load(root, "ana")
    assert p.roles == P.default_roles() and p.region == list(P.DEFAULT_REGION)


def test_command_line_refuses_a_pose_used_twice(tmp_path):
    with pytest.raises(SystemExit) as e:
        P.main(["ana", "--role", "right_click=pinch"], root=str(tmp_path))
    assert "twice" in str(e.value)
    assert P.load(str(tmp_path), "ana") is None                 # nothing was written
