"""Point-only control: hold still and a wheel of actions opens around the cursor.

For a hand that can point but cannot make a sign. The system cursor waits where it stopped while
the hand steers a choice by direction, so it works the same under any comfort-box gain. Pure:
update(xy, t, active) -> [Event], with the clock as a parameter. Distances are canonical view pixels.
"""
import math

from .fsm import DwellDetector, Event

ITEMS = ("click", "right", "drag", "scroll", "double", "pause")   # clockwise from the top
R_PICK, R_OUT = 16.0, 70.0          # past R_PICK a direction is being pointed at; past R_OUT the wheel closes
PICK_S, OPEN_TIMEOUT_S = 0.5, 6.0
CENTRE_S = 0.8                      # staying put clicks; long enough to start moving toward another item first
SCROLL_DEAD_PX, SCROLL_PX_PER_TICK, SCROLL_MAX_TICKS = 10.0, 15.0, 3
SCROLL_PERIOD_S, SCROLL_REST_S = 0.1, 1.2
EVENTS = {"click": "click", "right": "right_click", "double": "double_click", "pause": "pause"}


class Wheel:
    def __init__(self):
        self.dwell = DwellDetector()
        self.mode = "idle"          # idle | open | drag | scroll
        self.anchor, self.pick, self.progress = None, -1, 0.0
        self.t_open = self.t_pick = self.t_tick = self.t_rest = 0.0
        self.offset = (0.0, 0.0)

    @property
    def frozen(self):
        """True while the system cursor must stay put (the hand is steering the wheel or a scroll)."""
        return self.mode in ("open", "scroll")

    @property
    def view(self):
        """What to draw: mode, anchor, pointed item (-1 = centre), countdown 0..1, where the hand is steering."""
        return {"mode": self.mode, "anchor": self.anchor, "pick": self.pick, "progress": self.progress,
                "items": list(ITEMS), "dot": (self.offset[0] / R_OUT, self.offset[1] / R_OUT)}

    def reset(self):
        """Close without acting. -> the events needed to leave the system in a clean state."""
        events = [Event("drag_end")] if self.mode == "drag" else []
        self.mode, self.pick, self.progress = "idle", -1, 0.0
        self.dwell.reset()
        return events

    def _close(self, xy):
        self.mode, self.pick, self.progress = "idle", -1, 0.0
        self.dwell.reset()
        self.dwell.click_spot = xy          # re-arm only after the hand has moved away, as a dwell click does

    def update(self, xy, t, active):
        """`active` = the hand is in a neutral pose and the state machine is just moving the cursor."""
        if xy is None or not active:
            return self.reset()
        if self.mode == "idle":
            fired, self.progress = self.dwell.update(xy, t, True)
            if fired is not None:
                self.mode, self.anchor, self.offset = "open", xy, (0.0, 0.0)
                self.t_open, self.t_pick, self.pick, self.progress = t, t, -1, 0.0
            return []
        if self.mode == "drag":
            fired, self.progress = self.dwell.update(xy, t, True)
            if fired is None:
                return []
            self._close(xy)
            return [Event("drag_end")]
        self.offset = (xy[0] - self.anchor[0], xy[1] - self.anchor[1])
        return self._choose(xy, t) if self.mode == "open" else self._scroll(xy, t)

    def _choose(self, xy, t):
        dx, dy = self.offset
        r = math.hypot(dx, dy)
        if r > R_OUT or t - self.t_open > OPEN_TIMEOUT_S:
            self._close(xy)
            return []
        pick = -1 if r < R_PICK else int(((90.0 - math.degrees(math.atan2(-dy, dx))) % 360.0 + 30.0) // 60.0) % 6
        if pick != self.pick:
            self.pick, self.t_pick = pick, t
        self.progress = min((t - self.t_pick) / (CENTRE_S if pick == -1 else PICK_S), 1.0)
        if self.progress < 1.0:
            return []
        item = ITEMS[0] if pick == -1 else ITEMS[pick]
        if item == "drag":
            self.mode, self.progress = "drag", 0.0
            self.dwell.reset()
            self.dwell.click_spot = xy      # the drop needs a move and a new stop
            return [Event("drag_start")]
        if item == "scroll":
            self.mode, self.t_tick, self.t_rest, self.progress = "scroll", t, t, 0.0
            return []
        self._close(xy)
        return [Event(EVENTS[item])]

    def _scroll(self, xy, t):
        dy = self.offset[1]
        if abs(dy) <= SCROLL_DEAD_PX:
            self.progress = min((t - self.t_rest) / SCROLL_REST_S, 1.0)
            if self.progress >= 1.0:
                self._close(xy)
            return []
        self.t_rest, self.progress = t, 0.0
        if t - self.t_tick < SCROLL_PERIOD_S:
            return []
        self.t_tick = t
        ticks = min(int((abs(dy) - SCROLL_DEAD_PX) / SCROLL_PX_PER_TICK) + 1, SCROLL_MAX_TICKS)
        return [Event("scroll", ticks if dy < 0 else -ticks)]      # hand above the start point scrolls up
