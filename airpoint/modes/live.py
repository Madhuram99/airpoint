import time

import numpy as np

from .. import features as F
from .. import hud, panel
from ..actions import FOCUS_SETTLE_S
from ..fsm import Event
from ..face import FACE_LABELS, NEUTRAL
from ..helper import ASK_S, SHOW_S, Reins
from ..pointer import PRECLICK_RATIO
from ..poses import NONE, UNKNOWN



HEARD_S = 4.0     # how long the banner keeps what was heard
NOTE_S = 8.0      # how long it keeps how the AI helper's errand ended


def shake_line(sim, reading, left):
    """The message line's tremor read-out: what the meter sees, and what is left after the filters.
    None when there is nothing to say (no simulator, no lock)."""
    parts = []
    if sim.on:
        parts.append(f"Simulated tremor {sim.hz:.1f} Hz.")
    if reading.locked:
        parts.append(f"Locked on {reading.lock_hz:.1f} Hz: shake {reading.amp:.0f} px down to {left.amp:.1f} px.")
    elif sim.on:
        parts.append("Measuring...")
    return " ".join(parts) or None


def talk_line(talker, t):
    """The message line while talk is in use: listening, thinking, then what was heard for a moment."""
    if talker.state == "listening":
        how = ("Say a command or a sentence. L or the same sign ends it." if talker.local
               else "macOS dictation is typing what you say. The same sign ends it.")
        return f"Listening. {how} {talker.left(t):.0f} s left."
    if talker.state == "thinking":
        return "Working out what you said."
    if talker.t_end is not None and t - talker.t_end < HEARD_S and (talker.heard or talker.problem):
        return f'Heard "{talker.heard}".' if talker.heard else f"Talk: {talker.problem}."
    return None


def talk_tone(talker):
    if talker.state == "listening":
        return "act"
    return "warn" if talker.state == "idle" and talker.problem and not talker.heard else "track"


def helper_line(s, t):
    """The message line for the AI helper: what it is about to do, what it asks, how the errand ended."""
    h = s.helper
    if not h.active:
        recent = h.t_note is not None and t - h.t_note < NOTE_S and h.note
        return (("AI done: " if h.ok else "AI: ") + h.note) if recent else None
    v = h.view(t)
    head = f"AI step {v['n']} of {v['of']}: {v['text']}"
    if v["state"] == "asking":
        how = "Open mouth = yes, eyebrows = no" if s.source == "face" else "Thumb up = yes, thumb down = no"
        return f"{head} {how}, or point up or down. {max(ASK_S - (t - h.t_state), 0.0):.0f} s left."
    stop = "Eyebrows or a quick move stop it." if s.source == "face" else "A fist or a quick move stops it."
    if v["state"] == "showing":
        return f"{head} in {max(SHOW_S - (t - h.t_state), 0.0):.1f} s. {stop}"
    return f"{head}. {stop}"


def helper_overlay(s, view, events):
    """The glow while the AI helper works: on the step's target, saying what is about to happen there."""
    xy = None if view is None else (view["xy"] or s.runner.last_xy)
    if xy is None or view["hide"] or not s.runner.enabled:
        return {"show": False}                   # also hidden while the picture of the screen is taken
    asking, showing = view["state"] == "asking", view["state"] == "showing"
    state = {"show": True, "x": round(xy[0], 1), "y": round(xy[1], 1), "tone": "act",
             "ring": round(s.chooser.progress if asking else view["progress"] if showing else 0.0, 3),
             "tag": f"AI step {view['n']} of {view['of']}" + (": yes or no?" if asking else ""),
             "hint": view["text"][:44] or None}
    pulse = next((e.kind for e in events if e.kind in PULSES), None)
    if pulse:
        state["pulse"] = pulse
    return state


def brief(e):
    """An event as the banner shows it; a typed sentence is cut short."""
    return e.kind + ("" if e.arg is None else f" {e.arg}"[:40])


PULSES = ("click", "double_click", "right_click", "drag_start", "drag_end")
KEYED = ("type", "hotkey")      # events that are keystrokes: they land in whichever app has the keyboard


def overlay_state(s, o, smooth, progress, events):
    """One line for the cursor glow: where the real cursor is, what is happening there, and in which tone
    (track = following, act = an action, warn). Just after w or o was pressed it also shows what that key
    switched on, at the real cursor, even while the hand is not driving it."""
    demo = s.demo[0] if s.demo is not None and o.t < s.demo[1] else None
    preview = {"items": list(s.wheel.items), "pick": -1, "progress": 0.0, "preview": True}
    if not s.runner.enabled or smooth is None:
        if demo is None:
            return {"show": False}
        x, y = s.runner.where()
        state = {"show": True, "x": round(x, 1), "y": round(y, 1), "tone": "track"}
        if demo == "wheel":
            state["wheel"] = preview
        else:
            state["tag"] = "This ring follows your cursor"
        return state
    wheel = s.wheel.view if s.wheel_on else None
    at = wheel["anchor"] if wheel and s.wheel.frozen else smooth       # a frozen cursor waits at the anchor
    x, y = s.runner.to_screen(at[0] / o.w, at[1] / o.h)
    wheel_scroll = wheel is not None and wheel["mode"] == "scroll"
    scrolling = s.fsm.state == "SCROLLING" or wheel_scroll
    tag = ("Paused" if s.runner.locked else "Listening" if s.talker.state == "listening" else
           "Thinking" if s.talker.state == "thinking" else "Dragging" if s.runner.button_down else
           "Scrolling" if wheel_scroll else ("Scrolling up" if s.fsm.scroll_dir > 0 else "Scrolling down") if scrolling else
           "Simulated tremor" if s.sim.on else None)
    acting = s.runner.button_down or scrolling or s.fsm.state in ("PRESSING", "HOLD") or s.talker.state == "listening"
    tone = "warn" if s.runner.locked or tag == "Simulated tremor" else "act" if acting else "track"
    state = {"show": True, "x": round(x, 1), "y": round(y, 1), "ring": round(progress, 3), "tag": tag, "tone": tone,
             "hint": s.hint}
    pulse = next((e.kind for e in events if e.kind in PULSES), None)
    if pulse:
        state["pulse"] = pulse
    if wheel and wheel["mode"] == "open":
        state["wheel"] = {"items": wheel["items"], "pick": wheel["pick"], "progress": round(wheel["progress"], 3),
                          "dot": [round(v, 3) for v in wheel["dot"]]}
    elif demo == "wheel" and s.wheel_on:
        state["wheel"] = preview
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

    def _helped(self, s, o, frame, label, conf):
        """The AI helper has the cursor. The hand does not move it or click: it only stops or answers."""
        h, t = s.helper, o.t
        smooth = None
        if o.raw is not None:
            smooth = s.pointer.update(o.raw, t)
            hud.draw_cursor(frame, o.raw, smooth)
        asking = h.state == "asking"
        sign = s.reins.read(label, smooth, t, asking, s.source)
        events = []
        if sign in ("stop", "moved"):
            s.stop_helper(("you moved" if sign == "moved" else "you made the stop sign") + ": the helper stopped")
        elif sign in ("yes", "no"):
            events = h.answer(sign == "yes", t)
        elif asking:                                  # no sign for yes or no: point up or down instead
            steady = smooth is not None and Reins.steady(label, s.source)
            if steady and s.chooser.mode != "open":
                s.chooser.open_at(smooth, t)
            for e in s.chooser.update(smooth, t, steady):
                if e.kind == "answer":
                    events = h.answer(e.arg == "yes", t)
            hud.draw_wheel(frame, s.chooser.view)
        if h.active:
            events = events + h.tick(t, busy=s.runner.typing)
        if h.state != "asking":
            s.chooser.reset()
        s.runner.apply(events)
        for e in events:
            s.sounds.play(e.kind)
        view = h.view(t) if h.active else None
        if view is None and sign not in ("stop", "moved"):
            s.helper_ended()                          # it finished, failed or was refused in this frame
        s.hint = None
        if s.overlay is not None and s.overlay_on:
            s.overlay.send(helper_overlay(s, view, events))
        hud.draw_region(frame, s.runner.active_region, s.runner.enabled)
        s.lines = ((helper_line(s, t), "act"),)

    def tick(self, s, o, frame):
        seen, conf, preclick = (self._read_face if s.source == "face" else self._read_hand)(s, o)
        label = s.smoother.update(seen, conf)
        self.label = label
        if s.helper.active:
            return self._helped(s, o, frame, label, conf)
        acting = label
        if s.hands_off:                               # the sign that stopped or answered the helper is still up
            if label == NONE or (label != UNKNOWN and s.fsm.neutral(label)):
                s.hands_off = False
            else:
                acting = UNKNOWN
        state, events = s.fsm.update(acting, o.t)
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
        went, lost = None, False
        if s.runner.enabled and not s.runner.locked and any(e.kind in KEYED for e in events):
            somewhere, went = s.keys_home()          # words spoken from AirPoint's own window go to the app before it
            if not somewhere:
                events, lost = [e for e in events if e.kind not in KEYED], True
            elif went:
                events = [Event("settle", FOCUS_SETTLE_S)] + events
        s.runner.apply(events)
        if lost:
            s.last_event = "nowhere to type: click where the words should go first"
            s.say("Nowhere to type. Click where the words should go, then talk again.", o.t, "warn")
        if events:
            s.last_event = ", ".join(brief(e) for e in events if e.kind != "settle") + (f" -> {went}" if went else "")
            what = panel.did(events)
            if what and s.runner.enabled and not s.runner.locked:
                s.did = (what[:-1] + f", in {went}." if went else what, o.t)
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
                elif e.kind == "ask":
                    s.ask_helper(e.arg, o.t)
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
        hint = s.hint[0].upper() + s.hint[1:] + "." if s.hint and not s.hint.startswith("show your") else None
        s.lines = ((helper_line(s, o.t), "act"), (talk_line(s.talker, o.t), talk_tone(s.talker)), (hint, "warn"),
                   (shake, "warn" if s.sim.on else "track"))

    def on_key(self, s, key, t):
        return False

    def close(self, s):
        pass
