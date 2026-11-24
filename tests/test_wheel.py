from airpoint.wheel import ITEMS, Wheel

A = (300.0, 200.0)     # where the hand stops


def run(w, steps, tick=0):
    """steps: list of ((x, y) | None, seconds, active), fed at 10 Hz. -> (events as (kind, arg), next tick).
    Time is an integer tick count, so 10 steps are exactly one second."""
    out = []
    for xy, seconds, active in steps:
        for _ in range(round(seconds * 10)):
            out += [(e.kind, e.arg) for e in w.update(xy, tick / 10, active)]
            tick += 1
    return out, tick


def opened():
    w = Wheel()
    events, t = run(w, [(A, 1.1, True)])           # samples at 0.0 .. 1.0 s: the dwell completes on the last
    assert events == [] and w.mode == "open" and w.frozen and w.anchor == A
    return w, t


def at(i, r=30.0):
    """A point `r` px from the anchor in the direction of item i (clockwise from the top)."""
    import math
    a = math.radians(90.0 - 60.0 * i)
    return (A[0] + r * math.cos(a), A[1] - r * math.sin(a))


def test_holding_still_opens_the_wheel_and_staying_centred_clicks():
    w, t = opened()
    events, t = run(w, [(A, 0.6, True)], t)
    assert events == [] and 0.5 < w.progress < 1.0 and w.pick == -1
    events, t = run(w, [(A, 0.4, True)], t)
    assert events == [("click", None)] and w.mode == "idle" and not w.frozen


def test_each_direction_picks_its_item():
    expect = {"click": [("click", None)], "right": [("right_click", None)], "double": [("double_click", None)],
              "pause": [("pause", None)], "drag": [("drag_start", None)], "scroll": []}
    for i, item in enumerate(ITEMS):
        w, t = opened()
        events, t = run(w, [(at(i), 0.3, True)], t)
        assert events == [] and w.pick == i, item
        events, t = run(w, [(at(i), 0.4, True)], t)
        assert events == expect[item], item


def test_changing_direction_restarts_the_countdown():
    w, t = opened()
    events, t = run(w, [(at(1), 0.4, True), (at(2), 0.4, True)], t)
    assert events == [] and w.pick == 2
    events, t = run(w, [(at(2), 0.3, True)], t)
    assert events == [("drag_start", None)]


def test_moving_far_out_or_waiting_too_long_closes_it_without_acting():
    w, t = opened()
    events, t = run(w, [(at(1, 80.0), 0.2, True)], t)
    assert events == [] and w.mode == "idle"
    w, t = opened()
    wander = [(at(k % 6), 0.3, True) for k in range(24)]           # a new direction every 0.3 s: never settles
    events, t = run(w, wander, t)
    assert events == [] and w.mode == "idle"                        # six seconds passed: closed, nothing done


def test_it_does_not_reopen_until_the_hand_has_moved_away():
    w, t = opened()
    events, t = run(w, [(A, 0.9, True), (A, 3.0, True)], t)
    assert events == [("click", None)] and w.mode == "idle"            # still there for 3 s: no second wheel
    events, t = run(w, [((A[0] + 50, A[1]), 1.2, True)], t)
    assert w.mode == "open"


def test_drag_follows_the_hand_and_the_next_dwell_drops():
    w, t = opened()
    events, t = run(w, [(at(2), 0.7, True)], t)
    assert events == [("drag_start", None)] and w.mode == "drag" and not w.frozen
    events, t = run(w, [((400.0, 260.0), 0.5, True)], t)
    assert events == []
    events, t = run(w, [((400.0, 260.0), 0.7, True)], t)
    assert events == [("drag_end", None)] and w.mode == "idle"


def test_scroll_follows_hand_height_and_resting_in_the_middle_ends_it():
    w, t = opened()
    events, t = run(w, [(at(3), 0.7, True)], t)
    assert w.mode == "scroll" and w.frozen
    up, t = run(w, [((A[0], A[1] - 30.0), 0.5, True)], t)
    assert up and all(k == "scroll" and n > 0 for k, n in up)
    far, t = run(w, [((A[0], A[1] + 60.0), 0.5, True)], t)
    assert all(k == "scroll" and n < 0 for k, n in far) and max(-n for _, n in far) > max(n for _, n in up)
    rest, t = run(w, [(A, 1.0, True)], t)
    assert rest == [] and w.mode == "scroll"
    rest, t = run(w, [(A, 0.4, True)], t)
    assert rest == [] and w.mode == "idle"


def test_a_sign_or_a_lost_hand_closes_the_wheel_and_drops_a_drag():
    w, t = opened()
    events, t = run(w, [(A, 0.2, False)], t)                       # the user started making a sign
    assert events == [] and w.mode == "idle"
    w, t = opened()
    run(w, [(at(2), 0.7, True)], t)
    events, _ = run(w, [(None, 0.1, True)], t + 0.7)               # hand left the frame mid-drag
    assert events == [("drag_end", None)] and w.mode == "idle"


def test_view_describes_what_to_draw():
    w = Wheel()
    assert w.view["mode"] == "idle"
    w, t = opened()
    run(w, [(at(4), 0.2, True)], t)
    v = w.view
    assert v["mode"] == "open" and v["anchor"] == A and v["pick"] == 4 and v["items"] == list(ITEMS)
    assert 0.0 < v["progress"] < 1.0 and abs(v["dot"][0] ** 2 + v["dot"][1] ** 2 - (30.0 / 70.0) ** 2) < 1e-6
