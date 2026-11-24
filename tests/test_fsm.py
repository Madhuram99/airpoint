from airpoint.fsm import AirPointFSM
from airpoint.poses import NONE, UNKNOWN


def run(seq, fsm=None):
    fsm = fsm or AirPointFSM()
    out = []
    for pose, t in seq:
        _, events = fsm.update(pose, t)
        out += [(e.kind, e.arg) for e in events]
    return out, fsm


def test_short_pinch_is_a_click():
    events, fsm = run([("palm", 0.0), ("pinch", 0.1), ("pinch", 0.2), ("palm", 0.3)])
    assert events == [("click", None)]
    assert fsm.state == "MOVING"


def test_held_pinch_starts_and_ends_a_drag():
    events, _ = run([("palm", 0.0), ("pinch", 0.1), ("pinch", 0.65), ("pinch", 0.7), ("palm", 0.8)])
    assert events == [("drag_start", None), ("drag_end", None)]


def test_pinch_drifting_into_fist_keeps_the_drag():
    events, fsm = run([("palm", 0.0), ("pinch", 0.1), ("pinch", 0.7), ("fist", 0.8)])
    assert events == [("drag_start", None)]
    assert fsm.state == "DRAGGING"


def test_hand_lost_mid_drag_ends_it_and_rests():
    events, fsm = run([("palm", 0.0), ("pinch", 0.1), ("pinch", 0.7), (NONE, 0.8)])
    assert events == [("drag_start", None), ("drag_end", None)]
    assert fsm.state == "RESTING"


def test_unknown_holds_state():
    events, fsm = run([("palm", 0.0), ("pinch", 0.1), ("pinch", 0.7), (UNKNOWN, 0.8), (UNKNOWN, 0.9)])
    assert events == [("drag_start", None)]
    assert fsm.state == "DRAGGING"


def test_v_sign_fires_once_and_rearms_only_after_palm():
    events, _ = run([("palm", 0.0), ("v_sign", 0.1), ("v_sign", 0.45), ("v_sign", 0.6), ("v_sign", 0.9),
                     ("palm", 1.0), ("v_sign", 1.1), ("v_sign", 1.5)])
    assert events == [("right_click", None), ("right_click", None)]


def test_v_sign_released_early_does_nothing():
    events, fsm = run([("palm", 0.0), ("v_sign", 0.1), ("palm", 0.2)])
    assert events == []
    assert fsm.state == "MOVING"


def test_thumb_scrolls_at_fixed_rate_and_direction_follows_pose():
    events, fsm = run([("palm", 0.0), ("thumb_up", 0.1), ("thumb_up", 0.15), ("thumb_up", 0.2),
                       ("thumb_up", 0.26), ("thumb_down", 0.3), ("thumb_down", 0.4), ("palm", 0.5)])
    assert events == [("scroll", 2), ("scroll", 2), ("scroll", -2)]
    assert fsm.state == "MOVING"


def test_cursor_frozen_only_while_resting_or_scrolling():
    fsm = AirPointFSM()
    assert fsm.cursor_frozen
    fsm.update("palm", 0.0)
    assert not fsm.cursor_frozen
    fsm.update("thumb_up", 0.1)
    assert fsm.cursor_frozen


from airpoint.fsm import DwellDetector


def dwell(seq, d=None):
    d = d or DwellDetector()
    out = []
    for xy, t, active in seq:
        e, p = d.update(xy, t, active)
        out.append((e.kind if e else None, round(p, 2)))
    return out, d


def test_dwell_clicks_after_one_second_still():
    out, _ = dwell([((100, 100), 0.0, True), ((104, 98), 0.5, True), ((102, 101), 1.0, True)])
    assert out == [(None, 0.0), (None, 0.5), ("click", 1.0)]


def test_dwell_moving_restarts_the_timer():
    out, _ = dwell([((100, 100), 0.0, True), ((140, 100), 0.5, True),
                    ((141, 100), 1.2, True), ((141, 100), 1.5, True)])
    assert out[-2:] == [(None, 0.7), ("click", 1.0)]


def test_dwell_rearms_only_after_moving_30px():
    out, _ = dwell([((100, 100), 0.0, True), ((100, 100), 1.0, True),
                    ((100, 100), 3.0, True), ((120, 100), 4.5, True),
                    ((135, 100), 5.0, True), ((135, 100), 6.0, True)])
    assert [e for e, _ in out] == [None, "click", None, None, None, "click"]


def test_dwell_inactive_resets_progress():
    out, _ = dwell([((100, 100), 0.0, True), ((100, 100), 0.8, False),
                    ((100, 100), 0.9, True), ((100, 100), 1.5, True)])
    assert out == [(None, 0.0), (None, 0.0), (None, 0.0), (None, 0.6)]


def test_roles_can_be_remapped():
    from airpoint.fsm import Event
    fsm = AirPointFSM(hold_events={"fist": Event("right_click")}, grip={"pinch"})
    events, _ = run([("palm", 0.0), ("fist", 0.1), ("fist", 0.5), ("palm", 0.6),      # fist held = right click now
                     ("pinch", 0.7), ("palm", 0.9),                                    # pinch still clicks
                     ("v_sign", 1.0), ("v_sign", 1.5), ("palm", 1.6)], fsm)            # v_sign has no role any more
    assert events == [("right_click", None), ("click", None)]


def test_a_pose_with_no_role_is_neutral_it_moves_and_rearms():
    fsm = AirPointFSM(scroll={})                      # this user cannot make the thumb signs
    events, fsm = run([("palm", 0.0), ("thumb_up", 0.1), ("thumb_up", 0.5)], fsm)
    assert events == [] and fsm.state == "MOVING" and not fsm.cursor_frozen
    events, _ = run([("v_sign", 0.6), ("v_sign", 1.0), ("thumb_up", 1.1), ("v_sign", 1.2), ("v_sign", 1.6)], fsm)
    assert events == [("right_click", None), ("right_click", None)]    # thumb_up re-armed it, as palm would
