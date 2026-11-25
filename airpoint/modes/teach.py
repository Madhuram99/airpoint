import os

from .. import hud
from .. import profile as profiles
from ..calibration import Trainer
from ..recorder import Recorder, RecordSession, capture

TEACH_CAPTURE_S, MIN_FRAMES = 8.0, 60
ACTIONS = (("say", "speak", "phrase to speak"), ("hotkey", "keys", "keys, like cmd+c"),
           ("type", "type", "text to type"), ("open", "open", "app name or link"), ("pause", "pause", None),
           ("talk", "talk", None))
ENTER, ESC, BACKSPACE = (13, 10), 27, (8, 127)
WEAK_RECALL, MANY_FALSE_READS = 0.90, 0.02


class TeachMode:
    """`n`: name a sign, hold it for eight seconds, say what it does. The personal model is retrained
    with the new class and swapped in. With signs already taught, 1-9 on the name screen forgets one."""

    name = "teach"

    def __init__(self, s, t):
        self.path = os.path.join(s.data_dir, f"{s.user}_signs.csv")
        self.rec = Recorder(self.path, s.user)
        self.phase, self.text, self.error, self.message = "name", "", "", ""
        self.sign = self.spec = self.capture = self.trainer = self.action = None
        self.frames, self.forgot, self.exit = 0, None, False
        self.known = set(s.profile.signs)          # signs that existed before this visit to the screen

    @property
    def typing(self):
        return self.phase in ("name", "arg")

    @property
    def busy(self):
        return self.phase == "training"

    # ---- one frame
    def tick(self, s, o, frame):
        if self.phase == "name":
            taught = " | ".join(f"{i + 1} forgets {n}" for i, n in enumerate(list(s.profile.signs)[:9]))
            hud.draw_prompt(frame, "NEW SIGN: type a name", f"{self.text}_   Enter = next | Esc = cancel"
                            + (f" | {taught}" if taught and not self.text else ""), self.error or None)
        elif self.phase == "capture":
            phase, pose, left = self.capture.update(o.t)
            capture(self.capture, self.rec, pose, o.pts, o.hand, o.t)
            if self.capture.capturing and o.pts is not None:
                self.frames += 1
            if phase == "done":
                self.rec.close()
                if self.frames < MIN_FRAMES:
                    self._fail(f"only {self.frames} frames with a hand in view")
                else:
                    self.phase = "action"
            else:
                title = f"make {self.sign}   {left:.0f}" if phase == "countdown" else f"HOLD {self.sign}   {left:.0f}s"
                hud.draw_prompt(frame, title, f"tilt, move closer, move away | kept {self.frames}",
                                "no hand in view" if phase == "capturing" and o.pts is None else None)
        if self.phase == "action":
            menu = " | ".join(f"{i + 1} {label}" for i, (_, label, _) in enumerate(ACTIONS))
            hud.draw_prompt(frame, f"WHAT DOES {self.sign} DO?", menu + " | Esc = cancel")
        elif self.phase == "arg":
            hud.draw_prompt(frame, ACTIONS[self.action][2], f"{self.text}_   Enter = teach | Esc = cancel",
                            self.error or None)
        elif self.phase == "training":
            if self.trainer.done:
                self._finish(s)
            hud.draw_prompt(frame, "Learning...", "a few seconds")
        if self.phase in ("done", "failed"):
            hud.draw_prompt(frame, "done" if self.phase == "done" else "could not teach",
                            f"{self.message} | any key = back")

    # ---- keys
    def on_key(self, s, key, t):
        if self.busy:
            return True
        if key == ESC or self.phase in ("done", "failed"):
            self.exit = True
        elif self.phase == "name":
            self._name_key(s, key, t)
        elif self.phase == "action" and ord("1") <= key < ord("1") + len(ACTIONS):
            self.action = key - ord("1")
            if ACTIONS[self.action][2] is None:
                self._train(s, None)
            else:
                self.phase, self.text, self.error = "arg", "", ""
        elif self.phase == "arg":
            self._arg_key(s, key)
        return True                        # while teaching, every key belongs to the teach screen

    def _edit(self, key, limit):
        if key in BACKSPACE:
            self.text = self.text[:-1]
        elif 32 <= key <= 126 and len(self.text) < limit:
            self.text += chr(key)

    def _name_key(self, s, key, t):
        taught = list(s.profile.signs)
        if not self.text and ord("1") <= key <= ord("9") and key - ord("1") < len(taught):
            return self._forget(s, taught[key - ord("1")])
        if key not in ENTER:
            return self._edit(key, 20)
        name = self.text.strip().lower().replace(" ", "_")
        if not name:
            name = next(f"sign{i}" for i in range(1, 100) if f"sign{i}" not in s.profile.signs)
        self.error = profiles.check_sign_name(name, s.profile.signs) or ""
        if not self.error:
            self.sign, self.phase, self.frames = name, "capture", 0
            self.capture = RecordSession((name,), capture_s=TEACH_CAPTURE_S)
            self.capture.start(t)

    def _arg_key(self, s, key):
        if key not in ENTER:
            return self._edit(key, 80)
        kind, text = ACTIONS[self.action][0], self.text.strip()
        arg = profiles.parse_keys(text) if kind == "hotkey" else text
        if not arg:
            self.error = "not a key combo I know (try cmd+c, option+cmd+f5)" if kind == "hotkey" else "type something first"
        else:
            self._train(s, list(arg) if kind == "hotkey" else arg)

    # ---- training
    def _train(self, s, arg):
        self.spec = {"action": ACTIONS[self.action][0], "arg": arg, "hold_s": profiles.SIGN_HOLD_S}
        self._start(s)

    def _forget(self, s, name):
        self.rec.drop(name)
        self.rec.close()
        del s.profile.signs[name]
        self.forgot = name
        self._start(s)

    def _start(self, s):
        self.phase = "training"
        self.trainer = Trainer(lambda: s.train_fn(s.data_dir, s.models_dir, s.user))
        self.trainer.start()

    def _fail(self, why):
        self.phase, self.message = "failed", why
        if self.sign and self.sign not in self.known:           # leave no half-taught class behind
            rec = Recorder(self.path, self.rec.user)
            rec.drop(self.sign)
            rec.close()

    def _finish(self, s):
        trainer, self.trainer = self.trainer, None
        prof = s.profile
        if trainer.error and self.forgot and "usable frames of your own" in trainer.error:
            s.reset_model()                # nothing personal is left: back to the shared model
            prof.model = prof.labels = prof.acc_on_own_frames = None
            self.message = f"forgot {self.forgot}; the shared model is back"
        elif trainer.error:
            return self._fail(trainer.error)
        else:
            r = trainer.result
            s.use_model(r["model"], r["labels"])
            prof.model, prof.labels = os.path.relpath(r["model"], s.root), os.path.relpath(r["labels"], s.root)
            if self.forgot:
                self.message = f"forgot {self.forgot}"
            else:
                prof.signs[self.sign] = self.spec
                mine, false_reads = r["per_label"][self.sign], r["false_reads"].get(self.sign, 0.0)
                self.message = f"{self.sign}: {mine['recall'] * 100:.0f}% of its own frames recognised"
                if mine["recall"] < WEAK_RECALL and mine["confused_with"]:
                    self.message += f" | easily confused with {mine['confused_with']}"
                if false_reads > MANY_FALSE_READS:
                    self.message += f" | {false_reads * 100:.0f}% of other signs read as it: pick a more distinct sign"
        profiles.save(s.root, prof)
        s.apply_bindings()
        self.phase = "done"

    def close(self, s):
        self.rec.close()
