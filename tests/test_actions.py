import sys
import time

import pytest

from airpoint.actions import PG_KEYS, ActionRunner, FakeMouse, PyAutoGuiMouse, typable
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


def test_type_event_writes_text_through_the_mouse_object():
    m, r = make()
    r.apply([Event("type", "hello")])
    assert m.calls == []                              # mouse off: nothing is typed
    r.toggle()
    r.apply([Event("type", "hello"), Event("say", "spoken elsewhere")])
    assert m.calls == [("write", "hello")]


def test_double_click_event_is_one_os_double_click():
    m, r = make()
    r.toggle()
    r.apply([Event("double_click")])
    assert m.calls == [("double_click",)]


class SlowKeys:
    """Stands in for pyautogui: records what it is asked to press, as slowly as real keystrokes."""

    def __init__(self, broken=()):
        self.calls, self.broken = [], broken

    def hotkey(self, *keys, interval=0.0):
        time.sleep(0.05)
        if keys in self.broken:
            raise RuntimeError("no such key")
        self.calls.append(("hotkey",) + keys)

    def write(self, text, interval=0.0):
        time.sleep(0.2)
        self.calls.append(("write", text))


def test_keystrokes_queue_in_order_and_never_hold_up_the_frame_loop():
    pg = SlowKeys()
    m = PyAutoGuiMouse(pg)
    t0 = time.perf_counter()
    m.write("hello there")
    m.hotkey("control", "option", "d")
    assert time.perf_counter() - t0 < 0.05 and pg.calls == []
    assert m.flush(2.0)
    assert pg.calls == [("write", "hello there"), ("hotkey", "ctrl", "option", "d")]      # control is pyautogui's ctrl


def test_a_failing_keystroke_does_not_stop_the_ones_after_it(capsys):
    pg = SlowKeys(broken=(("f13",),))
    m = PyAutoGuiMouse(pg)
    m.hotkey("f13")
    m.hotkey("command", "c")
    assert m.flush(2.0) and pg.calls == [("hotkey", "command", "c")]
    assert "keystroke failed" in capsys.readouterr().out


def test_flush_reports_keystrokes_that_are_still_going():
    m = PyAutoGuiMouse(SlowKeys())
    m.write("a long sentence")
    assert m.flush(0.01) is False
    assert m.flush(2.0)


@pytest.mark.skipif(sys.platform != "darwin", reason="checks pyautogui's macOS key table")
def test_every_key_name_a_sign_may_use_is_one_pyautogui_can_press():
    osx = pytest.importorskip("pyautogui._pyautogui_osx")
    from airpoint.profile import KEY_NAMES
    assert [k for k in sorted(KEY_NAMES) if osx.keyboardMapping.get(PG_KEYS.get(k, k)) is None] == []


def test_typed_text_is_reduced_to_what_a_keyboard_can_type():
    assert typable("I’ll be late — sorry… “ok”") == "I'll be late - sorry... \"ok\""
    assert typable("café naïve now") == "cafe naive now"
    assert typable("line one\nline two") == "line one\nline two"
    assert typable("नमस्ते ok") == " ok"          # no key for these: left out


def test_the_runner_can_wait_for_keystrokes():
    m, r = make()
    assert r.flush() is True
