import math
from dataclasses import dataclass

from .poses import GRIP, NONE, SCROLL, UNKNOWN, V_SIGN

PRESS_HOLD_S = 0.50     # grip held this long -> drag instead of click
HOLD_FIRE_S = 0.30      # one-shot poses (v_sign, hotkeys) fire after this
SCROLL_PERIOD_S = 0.10
SCROLL_TICKS = 2
DWELL_S = 1.0           # still this long -> click
DWELL_STILL_PX = 15.0   # "still" = within this radius of where the cursor stopped
DWELL_REARM_PX = 30.0   # after a dwell click, move this far before the next one can arm


@dataclass(frozen=True)
class Event:
    kind: str          # click | right_click | double_click | drag_start | drag_end | scroll | hotkey | type | say | open | pause | talk
    arg: object = None  # scroll: signed ticks; hotkey: tuple of keys; type / say / open: text


DEFAULT_HOLD_EVENTS = {V_SIGN: Event("right_click")}


class AirPointFSM:
    """Pure: update(pose, t) -> (state, [Event]). Clock is a parameter."""

    def __init__(self, hold_events=None, grip=GRIP, scroll=None, hold_s=None):
        """Roles come from the user's profile: `grip` poses click and drag, `scroll` maps a pose to a
        direction, `hold_events` maps a pose to a one-shot event. A pose with no role is neutral:
        it moves the cursor and re-arms the one-shots, as the open palm does."""
        self.hold_events = dict(DEFAULT_HOLD_EVENTS if hold_events is None else hold_events)
        self.grip = frozenset(grip)
        self.scroll = dict(SCROLL if scroll is None else scroll)
        self.hold_s = dict(hold_s or {})     # per-pose hold time; HOLD_FIRE_S when a pose is not listed
        self.state = "RESTING"
        self.t_enter = 0.0
        self.hold_pose = None
        self.hold_fired = False
        self.hold_armed = True
        self.scroll_dir = 0
        self.next_scroll_t = 0.0

    def neutral(self, pose):
        return pose not in self.grip and pose not in self.scroll and pose not in self.hold_events

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
            if self.neutral(pose):
                self.hold_armed = True
            elif pose in self.grip:
                self._to("PRESSING", t)
            elif pose in self.scroll:
                self.scroll_dir, self.next_scroll_t = self.scroll[pose], t
                self._to("SCROLLING", t)
            elif pose in self.hold_events and self.hold_armed:
                self.hold_pose, self.hold_fired = pose, False
                self._to("HOLD", t)
        elif s == "PRESSING":
            held = t - self.t_enter
            if pose not in self.grip:
                if held < PRESS_HOLD_S:
                    events.append(Event("click"))
                self._to("MOVING", t)
            elif held >= PRESS_HOLD_S:
                events.append(Event("drag_start"))
                self._to("DRAGGING", t)
        elif s == "DRAGGING":
            if pose not in self.grip:
                events.append(Event("drag_end"))
                self._to("MOVING", t)
        elif s == "HOLD":
            if pose != self.hold_pose:
                if self.neutral(pose):
                    self.hold_armed = True
                self._to("MOVING", t)
            elif not self.hold_fired and t - self.t_enter >= self.hold_s.get(pose, HOLD_FIRE_S):
                events.append(self.hold_events[pose])
                self.hold_fired, self.hold_armed = True, False
        elif s == "SCROLLING":
            if pose not in self.scroll:
                self._to("MOVING", t)
            else:
                self.scroll_dir = self.scroll[pose]
                if t >= self.next_scroll_t:
                    events.append(Event("scroll", self.scroll_dir * SCROLL_TICKS))
                    self.next_scroll_t = t + SCROLL_PERIOD_S
        return self.state, events


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class DwellDetector:
    """Pure: update(xy, t, active) -> (Event | None, progress 0..1).

    Click when the cursor stays within `still_px` of its anchor for `dwell_s` while `active`.
    Disarmed after a click until the cursor moves `rearm_px` from the click spot.
    """

    def __init__(self, dwell_s=DWELL_S, still_px=DWELL_STILL_PX, rearm_px=DWELL_REARM_PX):
        self.dwell_s, self.still_px, self.rearm_px = dwell_s, still_px, rearm_px
        self.anchor, self.t_anchor, self.click_spot = None, 0.0, None

    def reset(self):
        self.anchor, self.click_spot = None, None

    def update(self, xy, t, active):
        if not active or xy is None:
            self.anchor = None
            return None, 0.0
        if self.click_spot is not None:
            if _dist(xy, self.click_spot) < self.rearm_px:
                self.anchor = None
                return None, 0.0
            self.click_spot = None
        if self.anchor is None or _dist(xy, self.anchor) > self.still_px:
            self.anchor, self.t_anchor = xy, t
            return None, 0.0
        progress = (t - self.t_anchor) / self.dwell_s
        if progress >= 1.0:
            self.click_spot, self.anchor = xy, None
            return Event("click"), 1.0
        return None, progress
