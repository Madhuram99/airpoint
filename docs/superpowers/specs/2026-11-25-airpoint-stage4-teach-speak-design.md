# AirPoint Stage 4 — Teach a sign, speak for me

Date: 2026-11-25
Status: approved in advance (user, 2026-11-25)
Roadmap: `2026-11-25-airpoint-v2-roadmap.md`, Stage 4

## 1. Goal

Anyone can add a sign their own hand can make and say what it does, in about
fifteen seconds, without touching code: speak a phrase aloud, press a key combo,
type text, open an app, or pause. This replaces the "two or three hotkey poses"
of the 13 November spec with as many as the user wants.

## 2. Flow — key `n` in live mode (needs `--user`)

1. Type a name, Enter. (Empty name = `sign1`, `sign2`, …) With signs already
   taught, `1`–`9` forgets one instead.
2. Three-second countdown, then hold the sign for eight seconds while tilting and
   moving it, as in recording.
3. Pick what it does: `1` speak, `2` keys, `3` type, `4` open, `5` pause; then type
   the phrase, the keys (`cmd+c`), the text, or the app name / link.
4. "Learning…" for a few seconds: the personal model is retrained off-thread with
   the new class and swapped in.
5. The result line gives the share of the sign's own held-out frames recognised,
   and a warning if it is easily confused with another sign.

Held for 0.6 s the sign fires once; it re-arms after a neutral pose, exactly like
the right-click sign.

## 3. Design

- **Data.** Custom frames go to `data/<user>_signs.csv` (ignored by git). The
  capture guard has no rule for an unknown label and lets it through.
- **Training.** `train_personal(data_dir, models_dir, user)` now gathers every
  personal file itself: `<user>_calib.csv` and `<user>_signs.csv`. Base rows plus
  personal rows ×3, mirrored as in Stage 3. It reports, per personal label, recall
  on a 25% hold-out and the label it is most confused with, and for each custom
  sign the share of the shared recordings wrongly read as that sign.
- **Bindings.** The profile gains `signs`: name → action, argument, hold time. The
  state machine already fires one-shot events for held poses; it gains a per-pose
  hold time. Sign names cannot reuse a built-in pose name.
- **Effects.** New event kinds beside the mouse ones: `say` (macOS `say`, voice from
  the profile; a new phrase interrupts the last), `hotkey`, `type`, `open`, `pause`.
  Typing, keys and opening need the mouse switch on; speaking works with it off,
  because a person who uses AirPoint to speak should not need mouse control. Nothing
  fires while paused.
- **Forgetting.** Removes the sign's rows and binding and retrains; with no personal
  rows left the shared model comes back.
- **X-ray.** With more than six classes the panel shows the six most probable.

## 4. Risks and what guards them

- *A passing hand shape read as the custom sign.* Majority vote over five frames at
  0.75 confidence, then a 0.6 s hold, then re-arm only after a neutral pose. The
  result line shows the measured false-read rate on the shared recordings and warns
  above 2%.
- *A sign too close to an existing one.* Recall under 90% on its own hold-out is
  reported with the label it is confused with.
- *Typing a name with `q` in it.* While a mode is taking text, the quit key is text.

## 5. Not here

An on-screen keyboard of our own: macOS has one, and a sign can be bound to the key
combo that opens Accessibility Shortcuts (`option+cmd+f5`). The wheel item for
custom actions arrives with the overlay in Stage 5.
