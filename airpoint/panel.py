"""What the side panel, the message line and the small view say. Pure: read from the session, draw nothing.

Words are for the person using the app, not for the code: Moving, not MOVING; "Pinch: let go to click",
not `pose pinch 0.98`. The numbers behind them are in the X-ray view.
"""
from .face import BROWS_UP, MOUTH_OPEN, NEUTRAL
from .poses import NONE, POSES, UNKNOWN

SIGN_NAMES = {"palm": "Open palm", "pinch": "Pinch", "fist": "Fist", "v_sign": "V sign",
              "thumb_up": "Thumb up", "thumb_down": "Thumb down"}
STATE_WORDS = {"RESTING": "Resting", "MOVING": "Moving", "PRESSING": "Pressing", "DRAGGING": "Dragging",
               "SCROLLING": "Scrolling"}
MODE_WORDS = {"record": ("Recording", "Follow the card over the picture. R goes back."),
              "calibrate": ("Calibrating", "Follow the card over the picture. C closes it."),
              "face_calibrate": ("Calibrating", "Follow the card over the picture. Esc cancels."),
              "teach": ("Teaching", "Follow the card over the picture. Esc cancels."),
              "ask": ("Ask the AI", "Type what you want done, then Enter. Esc cancels.")}
DID = {"click": "Clicked.", "right_click": "Right click.", "double_click": "Double click.", "drag_start": "Dragging.",
       "drag_end": "Dropped.", "pause": "Paused."}
TOAST_S, DID_S = 6.0, 3.0
MOUSE_OFF = "Mouse control is off. Press M and your hand moves the real cursor."


def name_of(sign):
    return SIGN_NAMES.get(sign) or sign.replace("_", " ").capitalize()


def does(event):
    """What a one-shot event does, in a few words."""
    kind, arg = event.kind, event.arg
    if kind == "right_click":
        return "right click"
    if kind == "say":
        return f'says "{arg}"'
    if kind == "hotkey":
        return "presses " + "+".join(arg)
    if kind == "type":
        return f'types "{arg}"'
    if kind == "open":
        return f"opens {arg}"
    return {"pause": "pauses AirPoint", "talk": "starts and stops talk", "ask": "asks the AI helper"}.get(kind, kind)


def role(fsm, sign):
    """-> (what the sign does, its tone) under the roles in use."""
    if sign in fsm.grip:
        return "click, hold to drag", "act"
    if sign in fsm.scroll:
        return "scroll up" if fsm.scroll[sign] > 0 else "scroll down", "act"
    if sign in fsm.hold_events:
        return does(fsm.hold_events[sign]), "act"
    return "move the cursor", "track"


def sign_rows(s, label):
    """One row per sign this user has: its drawing, its name, what it does, whether it is in view now."""
    if s.source == "face":
        usable = set(s.face_cfg["on"])
        rows = [dict(glyph="face:turn", name="Turn head", does="move the cursor", tone="track", lit=label == NEUTRAL),
                dict(glyph="face:mouth", name="Open mouth", does="click, hold to drag", tone="act",
                     lit=label == MOUTH_OPEN, off=MOUTH_OPEN not in usable),
                dict(glyph="face:brows", name="Raise brows", does="right click", tone="act",
                     lit=label == BROWS_UP, off=BROWS_UP not in usable),
                dict(glyph="face:still", name="Hold still", does="open the wheel" if s.wheel_on else "nothing yet",
                     tone="act", lit=False, off=not s.wheel_on)]
        return [dict(r, off=r.get("off", False)) for r in rows]
    rows = []
    for sign in list(POSES) + [n for n in s.profile.signs if n not in POSES]:
        what, tone = role(s.fsm, sign)
        off = sign in s.profile.skipped
        rows.append(dict(glyph=sign if sign in POSES else None, name=name_of(sign), does="not used" if off else what,
                         tone=tone, lit=label == sign and not off, off=off))
    return rows


def switch_rows(s):
    """Key, name and state of each switch. tone: track = on, act = busy right now, None = off."""
    def on(flag):
        return ("on", "track") if flag else ("off", None)

    talk = {"listening": ("listening", "act"), "thinking": ("thinking", "act")}.get(s.talker.state, ("", None))
    helper = ("working", "act") if s.helper.active else on(s.helper.armed)
    glow = on(s.overlay_on) if s.overlay is not None else ("none", None)
    rows = [("M", "Mouse", on(s.runner.enabled)), ("D", "Dwell", on(s.dwell_on)), ("W", "Wheel", on(s.wheel_on)),
            ("O", "Glow", glow), ("B", "Filter", (s.pointer.mode, "track" if s.pointer.mode != "off" else None)),
            ("F", "Face", on(s.source == "face")), ("L", "Talk", talk), ("A", "AI helper", helper),
            ("P", "Pause", ("paused", "act") if s.runner.locked else ("off", None)), ("?", "All keys", ("", None))]
    return [dict(key=key, name=name, value=value, tone=tone) for key, name, (value, tone) in rows]


def status(s):
    """-> (the one word, a line of detail, tone)."""
    if s.mode.name != "live":
        word, detail = MODE_WORDS.get(s.mode.name, (s.mode.name.capitalize(), ""))
        return word, detail, "idle"
    if s.helper.active:
        return "AI helper", "It has the cursor. A fist, a quick move or any key stops it.", "act"
    if s.runner.locked:
        return "Paused", "Nothing acts. Hold an open palm still for 2 s, or press P.", "warn"
    what = "face" if s.source == "face" else "hand"
    o, label, state = s.obs, s.mode.label, s.fsm.state
    seen = o is not None and (o.face is not None if what == "face" else o.pts is not None)
    if not seen or label == NONE:
        return f"No {what}", f"Show your {what} to the camera.", "idle"
    if label == UNKNOWN:
        return STATE_WORDS.get(state, "Moving"), "Sign not clear. Open your palm, then try again.", "idle"
    fsm = s.fsm
    if state == "HOLD":
        pose = fsm.hold_pose
        return name_of(pose) if what == "hand" else "Brows up", f"Hold it: {does(fsm.hold_events[pose])}.", "act"
    if state == "SCROLLING":
        return ("Scrolling up" if fsm.scroll_dir > 0 else "Scrolling down"), "Let go of the sign to stop.", "act"
    if state == "PRESSING":
        return "Pressing", "Let go to click. Keep holding to drag.", "act"
    if state == "DRAGGING":
        return "Dragging", "Move, then let go to drop.", "act"
    if s.wheel_on:
        detail = "Hold still for 1 s to open the wheel."
    elif s.dwell_on:
        detail = "Hold still for 1 s to click."
    elif what == "face":
        detail = "Turn your head to move. Open your mouth to click."
    else:
        detail = f"{name_of(label)} moves the cursor." if fsm.neutral(label) else f"{name_of(label)}: {role(fsm, label)[0]}."
    return STATE_WORDS.get(state, "Moving"), detail, "idle" if state == "RESTING" else "track"


def panel_model(s):
    word, detail, tone = status(s)
    label = s.mode.label if s.mode.name == "live" else None
    return dict(word=word, detail=detail, tone=tone, signs=sign_rows(s, label), switches=switch_rows(s))


def did(events):
    """What just happened, as the message line says it. '' for nothing worth a line."""
    for e in events:
        if e.kind in DID:
            return DID[e.kind]
        if e.kind == "scroll":
            return "Scrolling up." if e.arg > 0 else "Scrolling down."
        if e.kind == "hotkey":
            return "Pressed " + "+".join(e.arg) + "."
        if e.kind == "type":
            return f'Typed "{e.arg.strip()}".'
        if e.kind == "say":
            return f'Said "{e.arg}".'
        if e.kind == "open":
            return f"Opened {e.arg}."
    return ""


def bar_model(s, t, lines=()):
    """The message line. `lines` are (text, tone) offers from the mode, most pressing first."""
    if s.toast is not None and t - s.toast[2] < TOAST_S:
        return dict(text=s.toast[0], tone=s.toast[1])
    for text, tone in lines:
        if text:
            return dict(text=text, tone=tone)
    if s.mode.name == "live" and not s.runner.enabled:
        return dict(text=MOUSE_OFF, tone="idle")
    if s.did is not None and t - s.did[1] < DID_S:
        return dict(text=s.did[0], tone="act")
    return dict(text="", tone="idle")


def mini_model(s, t, lines=()):
    """The small view: the word, and the same line as the message line."""
    word, detail, tone = status(s)
    line = bar_model(s, t, lines)["text"]
    return dict(word=word, tone=tone, line=line or detail)
