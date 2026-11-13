from dataclasses import dataclass

from .poses import GRIP, NONE, PALM, SCROLL, UNKNOWN, V_SIGN

PRESS_HOLD_S = 0.50     # grip held this long -> drag instead of click
HOLD_FIRE_S = 0.30      # one-shot poses (v_sign, hotkeys) fire after this
SCROLL_PERIOD_S = 0.10
SCROLL_TICKS = 2


@dataclass(frozen=True)
class Event:
    kind: str          # click | right_click | drag_start | drag_end | scroll | hotkey
    arg: object = None  # scroll: signed ticks; hotkey: tuple of keys


DEFAULT_HOLD_EVENTS = {V_SIGN: Event("right_click")}


class AirPointFSM:
    """Pure: update(pose, t) -> (state, [Event]). Clock is a parameter."""

    def __init__(self, hold_events=None):
        self.hold_events = dict(DEFAULT_HOLD_EVENTS if hold_events is None else hold_events)
        self.state = "RESTING"
        self.t_enter = 0.0
        self.hold_pose = None
        self.hold_fired = False
        self.hold_armed = True
        self.scroll_dir = 0
        self.next_scroll_t = 0.0

    @property
    def cursor_frozen(self):
        return self.state in ("RESTING", "SCROLLING")

    def _to(self, state, t):
        self.state, self.t_enter = state, t

    def update(self, pose, t):
        events = []
        if pose == UNKNOWN:
            return self.state, events
        if pose == NONE:
            if self.state == "DRAGGING":
                events.append(Event("drag_end"))
            self._to("RESTING", t)
            return self.state, events

        s = self.state
        if s == "RESTING":
            self._to("MOVING", t)
        elif s == "MOVING":
            if pose == PALM:
                self.hold_armed = True
            elif pose in GRIP:
                self._to("PRESSING", t)
            elif pose in SCROLL:
                self.scroll_dir, self.next_scroll_t = SCROLL[pose], t
                self._to("SCROLLING", t)
            elif pose in self.hold_events and self.hold_armed:
                self.hold_pose, self.hold_fired = pose, False
                self._to("HOLD", t)
        elif s == "PRESSING":
            held = t - self.t_enter
            if pose not in GRIP:
                if held < PRESS_HOLD_S:
                    events.append(Event("click"))
                self._to("MOVING", t)
            elif held >= PRESS_HOLD_S:
                events.append(Event("drag_start"))
                self._to("DRAGGING", t)
        elif s == "DRAGGING":
            if pose not in GRIP:
                events.append(Event("drag_end"))
                self._to("MOVING", t)
        elif s == "HOLD":
            if pose != self.hold_pose:
                if pose == PALM:
                    self.hold_armed = True
                self._to("MOVING", t)
            elif not self.hold_fired and t - self.t_enter >= HOLD_FIRE_S:
                events.append(self.hold_events[pose])
                self.hold_fired, self.hold_armed = True, False
        elif s == "SCROLLING":
            if pose not in SCROLL:
                self._to("MOVING", t)
            else:
                self.scroll_dir = SCROLL[pose]
                if t >= self.next_scroll_t:
                    events.append(Event("scroll", self.scroll_dir * SCROLL_TICKS))
                    self.next_scroll_t = t + SCROLL_PERIOD_S
        return self.state, events
