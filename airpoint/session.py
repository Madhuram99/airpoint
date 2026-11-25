import json
import os
import time
from dataclasses import dataclass

import numpy as np

from . import features as F
from . import hud, panel, ui
from .actions import ACTIVE_REGION, FOCUS_SETTLE_S, Focus
from .classifier import PoseClassifier, VoteSmoother
from .effects import Effects
from .face import GRIP as FACE_GRIP, HOLD_EVENTS as FACE_HOLD, FacePointer, FaceSigns, default_face
from . import screen
from .fsm import AirPointFSM, DwellDetector, Event
from .helper import ASK_ITEMS, ASK_S, Helper, Reins, ready_check, thinker
from .hints import Hinter
from .modes import AskMode, CalibrateMode, FaceCalibrateMode, LiveMode, RecordMode, TeachMode
from .pointer import PointerPipeline
from .poses import NONE
from .profile import Profile, fsm_parts
from .sounds import Sounds
from .speech import make_talker
from .train import train_personal
from .tremor import TremorSim
from .wheel import Wheel


LIVE_EVERY_S = 1.0   # how often results/live.json is rewritten for the lab
FOCUS_EVERY_S = 0.3  # how often the app in front is looked at, to know where typed words should go
UNLOCK_S, UNLOCK_STILL_PX = 2.0, 20.0   # an open palm held this still, this long, ends a pause
FACE_GAIN = 1.6     # a head is a noisier pointer than a hand: smooth it as if it were a hand in a small box
FACE_CENTRE_FRAMES = 20   # uncalibrated face: the first frames set where "straight ahead" is
SIM_NUDGES = {ord("["): (0.0, -0.05), ord("]"): (0.0, 0.05), ord("{"): (-0.5, 0.0), ord("}"): (0.5, 0.0)}
ACTING = ("PRESSING", "DRAGGING", "HOLD", "SCROLLING")     # states in which a sign is doing something
DEMO_S = 2.2                    # how long the glow shows what w or o just switched on
FILTER_SAYS = {"full": "Filter: full. The shake is measured and cancelled, then the path is smoothed.",
               "smooth": "Filter: smoothing only. The shake is not cancelled.",
               "off": "Filter: off. The cursor follows the raw hand."}
NEEDS_NAME = "needs your name. Start AirPoint with --user <name>."


@dataclass
class Obs:
    """What one frame gave us."""
    t: float
    pts: object    # (21, 3) landmarks in normalized image coords, or None
    hand: object   # "Left" | "Right" | None
    raw: object    # pointer (x, y) in canonical view pixels (floats), or None
    w: int
    h: int
    fps: float
    luma: float = 255.0   # mean brightness of the view before anything is drawn on it
    face: object = None   # FaceObs when the face is the input, else None


class Session:
    """Everything the frame loop shares, minus camera, tracker and window (those stay in app.py).

    tick() runs one frame through the current mode and returns the image to show.
    on_key() handles a key: the mode gets it first, then the global keys.
    """

    def __init__(self, runner, rules, mlp, *, root, user=None, profile=None, dwell_on=None,
                 train_fn=train_personal, sounds=None, effects=None, overlay=None, talker=None, helper=None,
                 focus=None, theme=None, backing=1.0):
        self.runner, self.rules, self.mlp, self.base_mlp = runner, rules, mlp, mlp
        self.clf = mlp or rules
        self.root, self.user, self.train_fn = root, user, train_fn
        self.data_dir, self.models_dir = os.path.join(root, "data"), os.path.join(root, "models")
        self.profile = profile or Profile(user=user or "guest")      # a guest profile is never saved
        self.smoother, self.dwell = VoteSmoother(), DwellDetector()
        self.unlock_dwell = DwellDetector(dwell_s=UNLOCK_S, still_px=UNLOCK_STILL_PX)
        self.apply_bindings()
        self.sounds = sounds or Sounds(enabled=self.profile.sounds)
        self.effects = effects or Effects(voice=self.profile.voice)
        self.overlay, self.overlay_on = overlay, overlay is not None     # the cursor glow; None = none
        self.focus, self.focus_t = focus or Focus(), None                # the app the keyboard was in before ours
        self.talker = talker or make_talker(self.profile, press=lambda keys: runner.apply([Event("hotkey", keys)]))
        size = (runner.screen_w, runner.screen_h)
        self.helper = helper or Helper(thinker(), lambda: screen.shot(size, hide=screen.own_windows({os.getpid()})),
                                       size, check=ready_check)
        self.reins, self.chooser = Reins(), Wheel(items=ASK_ITEMS, centre=None, timeout=ASK_S + 5.0)
        self.hands_off = False          # after the helper: signs do nothing until a plain pointing pose is seen
        self.hinter, self.hints_on, self.hint = Hinter(), self.profile.hints, None
        self.pointer = PointerPipeline(self.profile.fc_min)
        self.pointer.mode = self.profile.filter
        self.apply_region(self.profile.region)
        self.sim = TremorSim()
        self.dwell_on = self.profile.dwell if dwell_on is None else dwell_on
        self.wheel, self.wheel_on = Wheel(), self.profile.wheel
        self.source, self.wheel_before_face = "hand", None           # "hand" | "face": what drives the pointer
        self.face_cfg = dict(self.profile.face) if self.profile.face else default_face()
        self.face_pointer, self.face_signs = FacePointer(self.face_cfg), FaceSigns(self.face_cfg)
        self.face_centring = None       # a list of (yaw, pitch) while the first look is being taken as the centre
        self.mini, self.view_dirty = False, False
        self.zoom, self.xray, self.ms_tracker = 1.25, False, 0.0
        self.theme, self.backing = theme or ui.DARK, float(backing)      # backing: real pixels per point of the display
        self.toast = self.did = None    # (sentence, tone, since) for the message line; (what just happened, since)
        self.lines, self.help, self._drawn = (), False, {}
        self.demo = None                # ("wheel" | "glow", until): the glow shows what a key just switched on
        self.recorder, self.rejected = None, 0
        self.mode = LiveMode()
        self.last_event, self.prev_t = "none", None
        self.obs, self.live_t = None, None

    def tick(self, frame, pts, hand, t, tracker_ms=0.0, face=None):
        self.ms_tracker = tracker_ms
        fps = 0.0 if self.prev_t is None else 1.0 / max(t - self.prev_t, 1e-5)
        self.prev_t = t
        if self.focus_t is None or not 0.0 <= t - self.focus_t < FOCUS_EVERY_S:
            self.focus_t = t
            self.focus.note()
        w, h = hud.canon_size(frame)                   # positions and thresholds stay in canonical units
        luma = hud.brightness(frame)
        k = self.zoom * self.backing
        view = hud.stage(frame, k)                     # the picture at the size it is shown
        raw = None
        if self.source == "face":
            pts, hand = None, None
            if face is not None:
                hud.draw_face(view, face)
                if self.face_centring is not None:
                    self._centre_face(face)
                else:
                    raw = self.face_pointer.xy(face.yaw, face.pitch, w, h)
        elif pts is not None:
            face = None
            if self.sim.on:
                pts = self.sim.apply(pts, t, w, h)
            hud.draw_skeleton(view, pts, lit=self.fsm.state in ACTING)
            raw = (float(pts[F.MIDDLE_MCP, 0] * w), float(pts[F.MIDDLE_MCP, 1] * h))
        self.obs = Obs(t, pts, hand, raw, w, h, fps, luma, face if self.source == "face" else None)
        self.lines = ()
        self.mode.tick(self, self.obs, view)
        self._write_live(t)
        live = self.mode.name == "live"
        if self.help:
            hud.draw_help(view)
        if live and self.mini:
            return hud.mini_frame(view, panel.mini_model(self, t, self.lines), self.theme, self.backing)
        if live and self.xray:
            side = hud.draw_xray(h, self.taps(), k, self.theme)
        else:
            side = self._drawn_once("side", (panel.panel_model(self), k, h),
                                    lambda key: hud.draw_panel(key[0], self.theme, k, h))
        width = (view.shape[1] + side.shape[1]) / k
        bar = self._drawn_once("bar", (panel.bar_model(self, t, self.lines), k, width),
                               lambda key: hud.draw_bar(key[0], self.theme, k, width))
        return hud.compose(view, side, bar, self.theme)

    def _drawn_once(self, name, key, draw):
        """The side panel and the message line change a few times a second, not every frame."""
        old = self._drawn.get(name)
        if old is None or old[0] != key:
            old = self._drawn[name] = (key, draw(key))
        return old[1]

    def say(self, text, t=None, tone="track"):
        """One sentence for the message line: what a key just did, or why it did nothing."""
        self.toast = (text, tone, (self.prev_t or 0.0) if t is None else t)

    def taps(self):
        """Everything the X-ray panel shows, read from where it already lives."""
        p, live = self.pointer, self.mode
        return dict(clf=self.clf.name, fps=self.obs.fps, ms_tracker=self.ms_tracker, ms_classifier=live.clf_ms,
                    labels=live.labels or self.clf.labels, probs=live.probs, gate=self.smoother.min_conf,
                    votes=[label for label, _ in self.smoother.buf], label=live.label,
                    state=self.fsm.state, state_s=self.obs.t - self.fsm.t_enter, last_event=self.last_event,
                    trail=p.trail, reading=p.reading, left=p.left, mode=p.mode, sim=self.sim.on)

    def apply_bindings(self):
        """The profile's roles and taught signs decide which pose does what. Restarts the state machine."""
        grip, scroll, hold, hold_s = fsm_parts(self.profile.roles, self.profile.signs)
        self.fsm = AirPointFSM(hold_events=hold, grip=grip, scroll=scroll, hold_s=hold_s)

    def _centre_face(self, face):
        """No calibration yet: wherever the head rests in its first frames becomes the middle of the screen,
        so someone sitting side-on to the camera is not pinned to an edge."""
        self.face_centring.append((face.yaw, face.pitch))
        if len(self.face_centring) >= FACE_CENTRE_FRAMES:
            yaw0, pitch0 = np.median(np.array(self.face_centring), axis=0)
            self.face_cfg = dict(self.face_cfg, yaw0=float(yaw0), pitch0=float(pitch0))
            self.face_pointer, self.face_centring = FacePointer(self.face_cfg), None

    def refresh_face(self):
        """Use the current face settings: pointer range, thresholds, and the face's roles in the state machine."""
        self.face_pointer, self.face_signs = FacePointer(self.face_cfg), FaceSigns(self.face_cfg)
        usable = set(self.face_cfg["on"])
        self.fsm = AirPointFSM(hold_events={k: e for k, e in FACE_HOLD.items() if k in usable},
                               grip=FACE_GRIP & usable, scroll={})

    def set_source(self, source):
        """Hand or face drives the pointer. Everything after the tracker stays the same."""
        if source == self.source:
            return
        self.stop_helper("switched input: the helper stopped")
        self.runner.apply(self.wheel.reset())
        self.runner.release()
        self.source, self.sim.on = source, False
        self.smoother.reset()
        if source == "face":
            self.refresh_face()
            self.face_centring = None if self.profile.face else []
            self.runner.active_region = (0.0, 0.0, 1.0, 1.0)      # the head range already spans the whole view
            self.pointer.set_gain(FACE_GAIN)
            self.wheel_before_face, self.wheel_on = self.wheel_on, True   # the wheel is how a face scrolls and drags
        else:
            self.apply_bindings()
            self.apply_region(self.profile.region)
            self.wheel_on = self.wheel_before_face

    def keys_home(self):
        """Keystrokes need somewhere to land. With AirPoint's own window in front, the app that was in
        front before it is brought forward. -> (there is somewhere, that app's name or None if nothing moved)."""
        if not self.focus.note():
            return True, None
        name = self.focus.give_back()
        return name is not None, name

    def toggle_talk(self, t):
        """One tap starts listening, the next ends it. What is heard is typed or carried out through the
        runner, so talk needs the mouse switch on and does not start during a pause."""
        if self.talker.state == "idle" and (not self.runner.enabled or self.runner.locked):
            self.last_event = "talk is off while paused" if self.runner.enabled else "talk needs mouse ON [m]"
            self.say("Talk is off while paused. Press P first." if self.runner.enabled
                     else "Talk needs mouse control. Press M first.", t, "warn")
            return
        if self.talker.state == "idle" and not self.talker.local:      # macOS types where the keyboard is
            somewhere, name = self.keys_home()
            if not somewhere:
                self.last_event = "talk: nowhere to type: click where the words should go first"
                self.say("Nowhere to type. Click where the words should go, then start talk again.", t, "warn")
                return
            if name:
                self.runner.apply([Event("settle", FOCUS_SETTLE_S)])
        was, now = self.talker.state, self.talker.toggle(t)
        if now == was:
            if self.talker.problem:
                self.last_event = f"talk: {self.talker.problem}"
                self.say(f"Talk: {self.talker.problem}.", t, "warn")
            return
        self.last_event = "talk: listening" if now == "listening" else "talk: stopped"
        if self.talker.local:                       # macOS dictation makes its own sounds
            self.sounds.play("talk_start" if now == "listening" else "talk_stop")

    def stop_talking(self):
        """Listening never outlives the mouse switch, a pause or the live view."""
        if self.talker.state == "listening":
            self.talker.stop(self.prev_t or 0.0)
            if self.talker.local:
                self.sounds.play("talk_stop")

    def heard_events(self, t):
        """Each frame: what a finished recognition asks for. Also ends a listen that ran too long."""
        was = self.talker.state
        events = self.talker.poll(t)
        if was == "listening" and self.talker.state != "listening" and self.talker.local:
            self.sounds.play("talk_stop")
        return events

    def ask_helper(self, task, t):
        """Hand an errand to the AI helper. It must have been switched on, and it acts through the runner."""
        if not self.helper.armed:
            self.last_event = "AI helper is off: a switches it on"
        elif not self.runner.enabled or self.runner.locked:
            self.last_event = "AI helper is off while paused" if self.runner.enabled else "AI helper needs mouse ON [m]"
        elif self.helper.active:
            self.last_event = "AI helper is busy"
        else:
            self.stop_talking()
            self.runner.apply(self.wheel.reset())
            self.runner.release()
            self.reins.reset()
            self.chooser.reset()
            if self.helper.start(task, t):
                self.last_event = f"asked the AI: {task}"[:60]
                return True
        return False

    def stop_helper(self, why):
        if self.helper.active:
            self.helper.stop(why, self.prev_t or 0.0)
            self.helper_ended()

    def helper_ended(self):
        """The errand is over, however it ended: say so, and give the cursor back from a clean state."""
        h = self.helper
        self.last_event = (("AI done: " if h.ok else "AI: ") + h.note)[:80]
        self.sounds.play("helper_done" if h.ok else "helper_stop")
        if h.ok:
            self.effects.say(h.note[:200])
        self.chooser.reset()
        self.dwell.reset()
        self.wheel.reset()
        self.runner.release()
        self.fsm.update(NONE, self.prev_t or 0.0)       # back to rest: nothing half-pressed carries over
        self.hands_off = True                           # the fist that stopped it must not become a click

    def set_locked(self, locked):
        """The user's pause. Locking lets go of a held button; a still open palm unlocks (live mode)."""
        if locked:
            self.stop_helper("paused: the helper stopped")
            self.stop_talking()
            self.wheel.reset()
            self.runner.lock()
        else:
            self.runner.unlock()
        self.unlock_dwell.reset()
        self.sounds.play("lock" if locked else "unlock")

    def apply_region(self, region):
        """Use this comfort box: it maps to the whole screen, and the pointer filters follow its gain."""
        self.runner.active_region = tuple(region)
        self.pointer.set_gain((ACTIVE_REGION[2] - ACTIVE_REGION[0]) / (region[2] - region[0]))

    def state(self):
        """What is driving the cursor right now, for the lab to label a test block with."""
        r = self.pointer.reading
        return dict(ts=time.time(), user=self.user, mode=self.mode.name, mouse=self.runner.enabled,
                    source=self.source, dwell=self.dwell_on, wheel=self.wheel_on, filter=self.pointer.mode,
                    sim=dict(on=self.sim.on, hz=self.sim.hz, amp=self.sim.amp),
                    locked=self.runner.locked, gain=round(self.pointer.gain, 3),
                    tremor_locked=r.locked, lock_hz=round(r.lock_hz, 2), classifier=self.clf.name,
                    talk=self.talker.state, helper=self.helper.state,
                    fps=round(self.obs.fps, 1) if self.obs else 0.0)

    def _write_live(self, t):
        if self.live_t is not None and t - self.live_t < LIVE_EVERY_S:
            return
        self.live_t = t
        path = os.path.join(self.root, "results", "live.json")
        try:                                   # the lab hook must never break the frame loop
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".tmp", "w") as f:
                json.dump(self.state(), f)
            os.replace(path + ".tmp", path)
        except OSError:
            pass

    def _hide_overlay(self):
        if self.overlay is not None and self.overlay_on:
            self.overlay.send({"show": False})

    def set_mode(self, mode):
        self.stop_helper("left the live view: the helper stopped")
        self.stop_talking()
        self.runner.apply(self.wheel.reset())       # a wheel drag does not survive leaving the live view
        self.mode.close(self)
        self.mode = mode
        if mode.name != "live":
            self._hide_overlay()
        if mode.name != "live":
            self.sim.on = False      # simulated shake never reaches recordings, calibration or a profile

    def use_model(self, model_path, labels_path):
        """Hot-swap to a personal model. Runs on the frame-loop thread."""
        self.mlp = self.clf = PoseClassifier.load(model_path, labels_path)
        self.mlp.name = f"mlp/{self.user}"
        self.smoother.reset()

    @property
    def typing(self):
        """True while a mode is taking text: every key, the quit key included, is then a character."""
        return getattr(self.mode, "typing", False)

    def reset_model(self):
        """Back to the shared model (or the rules), when nothing personal is left to use."""
        self.mlp = self.base_mlp
        self.clf = self.mlp or self.rules
        self.smoother.reset()

    def on_key(self, key, t):
        if self.helper.active:                      # the helper has the cursor: a key answers it or stops it
            if self.helper.state == "asking" and key in (ord("y"), 13, 10, ord("n")):
                self.runner.apply(self.helper.answer(key != ord("n"), t))
                if not self.helper.active:
                    self.helper_ended()
            else:
                self.stop_helper("a key was pressed: the helper stopped")
            return
        if self.mode.on_key(self, key, t):
            if getattr(self.mode, "exit", False):
                task = getattr(self.mode, "task", None)
                self.set_mode(LiveMode())
                if task:
                    self.ask_helper(task, t)
            return
        if self.source == "face" and key in (ord("r"), ord("n"), ord("t")):
            print("[AirPoint] record, teach and the tremor simulator are for the hand: press f first")
            self.say("Record, teach and the simulated tremor are for the hand. Press F first.", t, "warn")
            return
        live = self.mode.name == "live"
        if key == ord("m"):
            self.stop_talking()
            self.runner.toggle()
            self.say("Mouse control is on. Your hand moves the real cursor: an open palm moves it, a pinch clicks."
                     if self.runner.enabled else "Mouse control is off. Nothing is clicked: practise freely.", t)
        elif key == ord("l") and live:
            self.toggle_talk(t)
        elif key == ord("a") and live:
            problem = None if self.helper.armed else self.helper.problem()
            if problem:
                self.last_event = "AI helper: " + problem
                print(f"[AirPoint] AI helper: {problem}")
                self.say(f"AI helper: {problem}.", t, "warn")
            else:
                self.runner.release()
                self.set_mode(AskMode(self))
        elif key == ord("d"):
            self.dwell_on = not self.dwell_on
            self.dwell.reset()
            self.say("Dwell click is on. Hold the cursor still for 1 s and it clicks: no sign needed."
                     if self.dwell_on else "Dwell click is off.", t)
        elif key == ord("w"):
            self.wheel_on = not self.wheel_on
            self.runner.apply(self.wheel.reset())
            self.demo = ("wheel", t + DEMO_S) if self.wheel_on else None
            self.say("The wheel is on. Hold still for 1 s and a menu opens at the cursor: point at a choice to pick it."
                     if self.wheel_on else "The wheel is off.", t)
        elif key == ord("k") and self.mlp is not None:
            self.clf = self.rules if self.clf is self.mlp else self.mlp
            self.smoother.reset()
            self.say("Reading signs with the trained network." if self.clf is self.mlp
                     else "Reading signs with the hand-written rules, for comparison.", t)
        elif key == ord("v"):
            self.mini, self.view_dirty = not self.mini, True
            self.say("Small view, always on top. V brings the full window back." if self.mini else "Full window.", t)
        elif key == ord("z"):
            self.zoom = hud.ZOOMS[(hud.ZOOMS.index(self.zoom) + 1) % len(hud.ZOOMS)]
            self.view_dirty = True
            self.say(f"Zoom {self.zoom:g}x.", t)
        elif key == ord("x"):
            self.xray, self.view_dirty = not self.xray, True
            self.say("X-ray is on: what the system sees and decides, frame by frame." if self.xray else "X-ray is off.", t)
        elif key == ord("p"):
            self.set_locked(not self.runner.locked)
        elif key == ord("o") and self.overlay is not None:
            self._hide_overlay()
            self.overlay_on = not self.overlay_on
            self.demo = ("glow", t + DEMO_S) if self.overlay_on else None
            self.say(("The glow is on. A ring follows the real cursor and shows what AirPoint is doing."
                      + ("" if self.runner.enabled else " It appears once mouse control is on (M)."))
                     if self.overlay_on else "The glow is off.", t)
        elif key == ord("o"):
            self.say("No glow here: AirPoint was started without it.", t, "warn")
        elif key == ord("n") and live:
            if not self.user:
                print("[AirPoint] teaching a sign needs --user <name>")
                self.say(f"Teaching a sign {NEEDS_NAME}", t, "warn")
            else:
                self.runner.release()
                self.set_mode(TeachMode(self, t))
        elif key == ord("s"):
            self.sounds.enabled = not self.sounds.enabled
            self.say("Sounds are on." if self.sounds.enabled else "Sounds are off.", t)
        elif key == ord("h"):
            self.hints_on = not self.hints_on
            self.say("Hints are on." if self.hints_on else "Hints are off.", t)
        elif key == ord("b"):
            self.pointer.cycle_mode()
            self.say(FILTER_SAYS[self.pointer.mode], t)
        elif key == ord("f") and live:
            self.set_source("face" if self.source == "hand" else "hand")
            self.say("Face mode. Turn your head to move, open your mouth to click, raise your eyebrows for a right click."
                     if self.source == "face" else "Hand mode.", t)
        elif key == ord("t") and live:
            self.sim.on = not self.sim.on
            self.say("Simulated tremor is on, for the demo: the hand in the picture shakes and the filter has to cancel it."
                     if self.sim.on else "Simulated tremor is off.", t)
        elif key in SIM_NUDGES:
            self.sim.nudge(*SIM_NUDGES[key])
            self.say(f"Simulated tremor: {self.sim.hz:.1f} Hz, size {self.sim.amp:.2f} of the palm.", t)
        elif key == ord("r") and self.user and self.mode.name != "calibrate":
            self.runner.release()
            self.set_mode(RecordMode(self) if live else LiveMode())
        elif key == ord("r") and not self.user:
            self.say(f"Recording {NEEDS_NAME}", t, "warn")
        elif key == ord("c") and self.source == "face":
            if live:
                self.runner.release()
                self.set_mode(FaceCalibrateMode(self, t))
        elif key == ord("c") and self.mode.name != "record":
            if not self.user:
                print("[AirPoint] calibration needs --user <name>")
                self.say(f"Calibration {NEEDS_NAME}", t, "warn")
            elif live:
                self.runner.release()
                self.set_mode(CalibrateMode(self, t))
            elif not self.mode.busy:
                self.set_mode(LiveMode())
        elif key in (ord("?"), ord("/")):
            self.help = not self.help

    def close(self):
        self.stop_helper("AirPoint is closing")
        self.stop_talking()
        self.talker.close()
        self.runner.release()
        self.runner.flush()                         # the keys that end a dictation must leave before we do
        self.mode.close(self)
        if self.recorder is not None:
            self.recorder.close()
        if self.overlay is not None:
            self.overlay.close()
        try:
            os.remove(os.path.join(self.root, "results", "live.json"))
        except OSError:
            pass
