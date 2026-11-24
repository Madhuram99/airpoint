from .. import features as F
from .. import hud
from ..pointer import PRECLICK_RATIO


class LiveMode:
    """Classify -> vote -> state machine -> pointer -> dwell -> OS."""

    name = "live"

    def tick(self, s, o, frame):
        if o.pts is None:
            label, conf, ratio = s.smoother.update(None, 0.0), 0.0, 1.0
        else:
            raw_label, conf = s.clf.predict(o.pts)
            label, ratio = s.smoother.update(raw_label, conf), F.pinch_ratio(o.pts)
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
        sim = f"SIMULATED TREMOR {s.sim.hz:.1f} Hz, {s.sim.amp:.2f} palm" if s.sim.on else None
        s.mini_label = (f"{state} | {label} {conf:.2f} | mouse {'ON' if s.runner.enabled else 'OFF'}"
                        + (" | SIM" if sim else ""))
        if not s.mini:
            hud.draw_banner(
                frame,
                f"{state} | pose {label} {conf:.2f} | ratio {ratio:.2f} | {s.clf.name} | {o.fps:.0f} fps",
                f"last: {s.last_event} | mouse [m] {'ON' if s.runner.enabled else 'OFF'} | "
                f"dwell [d] {'ON' if s.dwell_on else 'OFF'} | {hud.KEY_HINT}",
                s.runner.enabled,
                line3=(sim, hud.ALERT) if sim else None,
            )

    def on_key(self, s, key, t):
        return False

    def close(self, s):
        pass
