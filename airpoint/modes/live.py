import time

import numpy as np

from .. import features as F
from .. import hud
from ..actions import in_front
from ..face import FACE_LABELS, NEUTRAL
from ..pointer import PRECLICK_RATIO
from ..poses import NONE, UNKNOWN


FILTER_NAMES = {"full": "full", "smooth": "smooth only", "off": "OFF"}
HEARD_S = 4.0     # how long the banner keeps what was heard


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


def talk_line(talker, t):
    """Third banner line while talk is in use: listening, thinking, then what was heard for a moment."""
    if talker.state == "listening":
        how = "say a command or a sentence" if talker.local else "macOS dictation is typing what you say"
        return f"LISTENING [l]: {how} | the same sign or l ends it | {talker.left(t):.0f} s"
    if talker.state == "thinking":
        return "THINKING: working out what you said"
    if talker.t_end is not None and t - talker.t_end < HEARD_S and (talker.heard or talker.problem):
        return f'heard "{talker.heard}"' if talker.heard else f"talk: {talker.problem}"
    return None


def brief(e):
    """An event as the banner shows it; a typed sentence is cut short."""
    return e.kind + ("" if e.arg is None else f" {e.arg}"[:40])


PULSES = ("click", "double_click", "right_click", "drag_start", "drag_end")


def overlay_state(s, o, smooth, progress, events):
    """One line for the cursor glow: where the real cursor is and what to show around it."""
    if not s.runner.enabled or smooth is None:
        return {"show": False}
    wheel = s.wheel.view if s.wheel_on else None
    at = wheel["anchor"] if wheel and s.wheel.frozen else smooth       # a frozen cursor waits at the anchor
    x, y = s.runner.to_screen(at[0] / o.w, at[1] / o.h)
    scrolling = s.fsm.state == "SCROLLING" or (wheel is not None and wheel["mode"] == "scroll")
    tag = ("PAUSED" if s.runner.locked else "LISTENING" if s.talker.state == "listening" else
           "THINKING" if s.talker.state == "thinking" else "DRAG" if s.runner.button_down else
           "SCROLL" if scrolling else "SIMULATED" if s.sim.on else None)
    state = {"show": True, "x": round(x, 1), "y": round(y, 1), "ring": round(progress, 3), "tag": tag, "hint": s.hint}
    pulse = next((e.kind for e in events if e.kind in PULSES), None)
    if pulse:
        state["pulse"] = pulse
    if wheel and wheel["mode"] == "open":
        state["wheel"] = {"items": wheel["items"], "pick": wheel["pick"], "progress": round(wheel["progress"], 3),
                          "dot": [round(v, 3) for v in wheel["dot"]]}
    return state


class LiveMode:
    """Classify -> vote -> state machine -> pointer -> dwell -> OS."""

    name = "live"

    def __init__(self):
        self.probs, self.label, self.clf_ms = None, "none", 0.0     # kept for the X-ray panel
        self.labels, self.detail = [], ""

    def _read_hand(self, s, o):
        """-> (label this frame | None, confidence, about to click?)."""
        self.labels = s.clf.labels
        if o.pts is None:
            self.probs, self.detail = None, "ratio 1.00"
            return None, 0.0, False
        t0 = time.perf_counter()
        self.probs = s.clf.proba(o.pts)
        self.clf_ms = (time.perf_counter() - t0) * 1000.0
        i, ratio = int(np.argmax(self.probs)), F.pinch_ratio(o.pts)
        self.detail = f"ratio {ratio:.2f}"
        return s.clf.labels[i], float(self.probs[i]), ratio < PRECLICK_RATIO

    def _read_face(self, s, o):
        self.labels, self.clf_ms = list(FACE_LABELS), 0.0
        if o.face is None:
            self.probs, self.detail = None, "no face"
            return None, 0.0, False
        label, self.probs, near = s.face_signs.update(o.face.shapes)
        if s.talker.mouth_busy(o.t):                 # a speaking face is talking, not making signs
            self.detail = "talking"
            return NEUTRAL, 1.0, False
        self.detail = f"mouth {near:.2f}"
        return label, 1.0, label == NEUTRAL and near > 0.5

    def tick(self, s, o, frame):
        seen, conf, preclick = (self._read_face if s.source == "face" else self._read_hand)(s, o)
        label = s.smoother.update(seen, conf)
        self.label = label
        state, events = s.fsm.update(label, o.t)
        smooth = None
        if o.raw is not None:
            smooth = s.pointer.update(o.raw, o.t, preclick=preclick)
            hud.draw_cursor(frame, o.raw, smooth)
            if not s.fsm.cursor_frozen and not (s.wheel_on and s.wheel.frozen):
                s.runner.move(smooth[0] / o.w, smooth[1] / o.h)
        if s.runner.locked:                        # paused: nothing acts; a still open palm resumes
            events = []
            calm = smooth is not None and label not in (NONE, UNKNOWN) and s.fsm.neutral(label)
            resume, progress = s.unlock_dwell.update(smooth, o.t, calm)
            if resume is not None:
                s.set_locked(False)
        elif s.wheel_on:                           # point-only: a dwell opens the wheel of actions
            events = events + s.wheel.update(smooth, o.t, state == "MOVING")
            progress = 0.0 if s.wheel.mode == "open" else s.wheel.progress     # the open wheel draws its own countdown
            hud.draw_wheel(frame, s.wheel.view)
        else:
            dwell_event, progress = s.dwell.update(smooth, o.t, s.dwell_on and state == "MOVING")
            if dwell_event is not None:
                events = events + [dwell_event]
        if any(e.kind == "talk" for e in events):
            s.toggle_talk(o.t)
        events = [e for e in events if e.kind != "talk"] + s.heard_events(o.t)
        if smooth is not None:
            hud.draw_dwell_ring(frame, smooth, progress)
        s.runner.apply(events)
        if events:
            s.last_event = ", ".join(brief(e) for e in events)
            if s.runner.enabled and any(e.kind == "type" for e in events) and in_front():
                s.last_event = "nowhere to type: click where the words should go first"
            if s.runner.enabled:
                for e in events:
                    s.sounds.play(e.kind)
            for e in events:                           # what a taught sign can do beyond the mouse
                if e.kind == "say":
                    s.effects.say(e.arg)               # speech does not need mouse control
                elif e.kind == "open" and s.runner.enabled:
                    s.effects.open(e.arg)
                elif e.kind == "pause" and not s.runner.locked:
                    s.set_locked(True)
        hud.draw_region(frame, s.runner.active_region, s.runner.enabled)
        s.hint = None
        if s.hints_on:
            centre = (0.5, 0.5) if o.raw is None else (o.raw[0] / o.w, o.raw[1] / o.h)
            if s.source == "face":                   # a face has no "too far" yet: only presence, light, clarity
                s.hint = s.hinter.update(o.t, o.luma, None if o.face is None else 70.0, centre, label, o.fps, "face")
            else:
                s.hint = s.hinter.update(o.t, o.luma, None if o.pts is None else F.palm_px(o.pts, o.w, o.h),
                                         centre, label, o.fps)
        if s.overlay is not None and s.overlay_on:
            s.overlay.send(overlay_state(s, o, smooth, progress, events))
        shake = shake_line(s.sim, s.pointer.reading, s.pointer.left)
        talk = talk_line(s.talker, o.t)
        if s.runner.locked:
            line3 = ("PAUSED [p]: hold an open palm still for 2 s to resume", hud.ACCENT_ON)
        elif talk:
            line3 = (talk, hud.ACCENT_ON)
        else:
            line3 = (shake, hud.ALERT if s.sim.on else hud.MINT) if shake else None
        s.mini_label = ("! " + s.hint) if s.hint else (
            ("PAUSED | " if s.runner.locked else "")
            + ("" if s.talker.state == "idle" else s.talker.state.upper() + " | ")
            + f"{state} | {label} {conf:.2f} | mouse {'ON' if s.runner.enabled else 'OFF'}"
            + (" | SIM" if s.sim.on else "")
            + (f" | LOCK {s.pointer.reading.lock_hz:.1f}Hz" if s.pointer.reading.locked else ""))
        if not s.mini:
            hud.draw_banner(
                frame,
                f"{state} | pose {label} {conf:.2f} | {self.detail} | "
                f"{'face' if s.source == 'face' else s.clf.name} | {o.fps:.0f} fps",
                f"last: {s.last_event} | mouse [m] {'ON' if s.runner.enabled else 'OFF'} | "
                f"dwell [d] {'ON' if s.dwell_on else 'OFF'} | wheel [w] {'ON' if s.wheel_on else 'OFF'} | "
                f"filter [b] {FILTER_NAMES[s.pointer.mode]}",
                s.runner.enabled,
                line3=line3,
            )
            if s.hint:
                hud.draw_hint(frame, s.hint)

    def on_key(self, s, key, t):
        return False

    def close(self, s):
        pass
