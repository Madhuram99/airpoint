import time

import numpy as np

from .. import features as F
from .. import hud
from ..pointer import PRECLICK_RATIO


FILTER_NAMES = {"full": "full", "smooth": "smooth only", "off": "OFF"}


def shake_line(sim, reading, left):
    """Third banner line: what the tremor meter sees, and what is left after the filters.
    None when there is nothing to say (no simulator, no lock)."""
    parts = []
    if sim.on:
        parts.append(f"SIMULATED TREMOR {sim.hz:.1f} Hz")
    if reading.locked:
        parts.append(f"LOCKED {reading.lock_hz:.1f} Hz: shake {reading.amp:.0f} px -> {left.amp:.1f} px")
    elif sim.on:
        parts.append("measuring...")
    return " | ".join(parts) or None


class LiveMode:
    """Classify -> vote -> state machine -> pointer -> dwell -> OS."""

    name = "live"

    def __init__(self):
        self.probs, self.label, self.clf_ms = None, "none", 0.0     # kept for the X-ray panel

    def tick(self, s, o, frame):
        if o.pts is None:
            self.probs, label, conf, ratio = None, s.smoother.update(None, 0.0), 0.0, 1.0
        else:
            t0 = time.perf_counter()
            self.probs = s.clf.proba(o.pts)
            self.clf_ms = (time.perf_counter() - t0) * 1000.0
            i = int(np.argmax(self.probs))
            conf = float(self.probs[i])
            label, ratio = s.smoother.update(s.clf.labels[i], conf), F.pinch_ratio(o.pts)
        self.label = label
        state, events = s.fsm.update(label, o.t)
        smooth = None
        if o.raw is not None:
            smooth = s.pointer.update(o.raw, o.t, preclick=ratio < PRECLICK_RATIO)
            hud.draw_cursor(frame, o.raw, smooth)
            if not s.fsm.cursor_frozen:
                s.runner.move(smooth[0] / o.w, smooth[1] / o.h)
        dwell_event, progress = s.dwell.update(smooth, o.t, s.dwell_on and state == "MOVING")
        if dwell_event is not None:
            events = events + [dwell_event]
        if smooth is not None:
            hud.draw_dwell_ring(frame, smooth, progress)
        s.runner.apply(events)
        if events:
            s.last_event = ", ".join(e.kind + ("" if e.arg is None else f" {e.arg}") for e in events)
        hud.draw_region(frame, s.runner.active_region, s.runner.enabled)
        shake = shake_line(s.sim, s.pointer.reading, s.pointer.left)
        s.mini_label = (f"{state} | {label} {conf:.2f} | mouse {'ON' if s.runner.enabled else 'OFF'}"
                        + (" | SIM" if s.sim.on else "")
                        + (f" | LOCK {s.pointer.reading.lock_hz:.1f}Hz" if s.pointer.reading.locked else ""))
        if not s.mini:
            hud.draw_banner(
                frame,
                f"{state} | pose {label} {conf:.2f} | ratio {ratio:.2f} | {s.clf.name} | {o.fps:.0f} fps",
                f"last: {s.last_event} | mouse [m] {'ON' if s.runner.enabled else 'OFF'} | "
                f"dwell [d] {'ON' if s.dwell_on else 'OFF'} | filter [b] {FILTER_NAMES[s.pointer.mode]} | "
                f"{hud.KEY_HINT}",
                s.runner.enabled,
                line3=(shake, hud.ALERT if s.sim.on else hud.MINT) if shake else None,
            )

    def on_key(self, s, key, t):
        return False

    def close(self, s):
        pass
