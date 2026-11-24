from airpoint.actions import ActionRunner, FakeMouse
from airpoint.fsm import Event


def make():
    m = FakeMouse()
    return m, ActionRunner(m, m.size())


def test_disabled_runner_moves_nothing_and_applies_nothing():
    m, r = make()
    r.move(0.5, 0.5)
    r.apply([Event("click"), Event("scroll", 2)])
    assert m.calls == []


def test_active_region_maps_to_full_screen_and_clamps():
    _, r = make()
    assert r.to_screen(0.0, 0.0) == (0.0, 0.0)
    assert r.to_screen(1.0, 1.0) == (999.0, 499.0)
    sx, sy = r.to_screen(0.5, 0.475)
    assert abs(sx - 499.5) < 1e-6 and abs(sy - 249.5) < 1e-6


def test_events_map_to_mouse_calls_when_enabled():
    m, r = make()
    r.toggle()
    r.move(0.15, 0.15)
    r.apply([Event("click"), Event("right_click"), Event("scroll", -2),
             Event("hotkey", ("command", "tab"))])
    assert m.calls == [("move", 0, 0), ("click",), ("right_click",), ("scroll", -2),
                       ("hotkey", "command", "tab")]


def test_drag_end_without_drag_start_is_a_noop():
    m, r = make()
    r.toggle()
    r.apply([Event("drag_end")])
    assert m.calls == []


def test_toggle_off_releases_a_held_button():
    m, r = make()
    r.toggle()
    r.apply([Event("drag_start")])
    r.toggle()
    assert m.calls == [("down",), ("up",)]
    assert not r.enabled and not r.button_down


def test_a_locked_runner_sends_nothing_and_lets_go_of_the_button():
    m, r = make()
    r.toggle()
    r.apply([Event("drag_start")])
    r.lock()
    assert r.locked and m.calls == [("down",), ("up",)]
    r.move(0.5, 0.5)
    r.apply([Event("click"), Event("scroll", 2)])
    assert m.calls == [("down",), ("up",)]
    r.unlock()
    r.apply([Event("click")])
    assert m.calls[-1] == ("click",)
