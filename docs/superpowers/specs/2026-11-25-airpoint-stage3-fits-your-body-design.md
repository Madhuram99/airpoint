# AirPoint Stage 3 — Fits your body

Date: 2026-11-25
Status: approved in advance (user, 2026-11-25)
Roadmap: `2026-11-25-airpoint-v2-roadmap.md`, Stage 3

## 1. Goal

AirPoint adapts to the person instead of asking the person to adapt: the reach
they find comfortable, the signs their hand can make, whichever hand they use, a
way to pause without leaving the camera, and plain-language hints when something
is wrong.

## 2. Facts measured first

- On the recorded frames mirrored left-to-right (what the other hand looks like)
  the shipped model is right 64.8% of the time, against 99.6% unmirrored.
- MediaPipe's own Left/Right label is not a hand identity: in the recordings it
  says Right for palm, pinch and fist and Left for the V-sign and both thumbs of
  the same hand, because it follows which way the hand faces.
- The recorded hand is 54–87 px long (wrist to middle knuckle) in the 640-wide
  view and used only x 0.29–0.75, y 0.32–0.88 of the frame.

## 3. Design

### 3.1 Profile v2 — `profile.py`
One JSON per user holds everything personal: model paths, hold-out score,
`fc_min`, comfort box, roles, skipped poses, dwell, sounds, hints, filter mode.
Version 1 files (Week 2) are upgraded on load. `python -m airpoint.profile <user>`
shows the profile and edits roles, the box and the switches, refusing a role map
that uses one pose twice.

### 3.2 Choose your poses — roles
The state machine stops importing fixed sets. It is given roles: which poses grip
(click, drag), which scroll up and down, which fire a held action (right click).
Any pose with no role is neutral: it moves the cursor and re-arms held actions,
as the open palm does today. Defaults reproduce today's behaviour exactly.

Calibration notices what a hand cannot do: a pose whose capture kept under 30% of
its frames is marked skipped and loses its role. If no grip pose is left, dwell
click is switched on, because otherwise the user could not click at all.

### 3.3 Either hand
Training adds a left-to-right mirrored copy of every example, for the shared model
and for personal ones. One model then serves both hands and both faces of a hand,
with no use of MediaPipe's handedness label. The training report gains the accuracy
on mirrored hold-out frames, so "works with either hand" has a number. The shipped
model is retrained.

### 3.4 Comfort box
Calibration gains a reach step: six seconds of "move your hand as far as is
comfortable, in every direction". The box is the 5th–95th percentile of the
pointer, padded by 10%, at least a quarter of the frame each way. It replaces the
fixed active region and is drawn on the view as today.

A small box means high gain: the same shake travels further on screen. The pointer
pipeline is told the gain relative to the default box, lowers the smoothing cutoff
by it (not below 0.3 Hz) and raises the 1-Euro speed gain by it, so cursor feel
stays comparable. With the default box nothing changes.

### 3.5 Pause
`p` locks AirPoint: nothing reaches the operating system and a held button is
released. Unlock without a keyboard by holding an open palm still for two seconds;
the ring shows the countdown. (A held fist cannot be the trigger: it is drag.)
The lock becomes a wheel item and a bindable sign in Stages 4 and 5.

### 3.6 Hints and sounds
`hints.py` turns what the frame shows into one short instruction, most urgent
first: show your hand, too dark, move closer, move back, move toward the middle,
sign not clear, camera slow. It appears as a strip at the bottom of the view and in
the mini label. `sounds.py` plays a system sound for click, right click, drag
start, drag end, lock and unlock, only when the action really reached the system.
Keys `h` and `s` switch them.

## 4. Tests
Profile round trip, v1 upgrade and role validation; state machine with remapped
and missing roles; mirrored hands classify like unmirrored after training; comfort
box geometry and the gain it implies; lock gating and the still-palm unlock; each
hint's threshold and their priority; the sound for each event; calibration with
the reach step, the skip rule and the dwell fallback; launch applies a profile.

## 5. Limits
The gain compensation is reasoned, not tuned on a camera. The skip rule uses the
capture guard, which is deliberately loose. Role editing is a command, not a
screen; a visual chooser can ride on the Stage 5 overlay.
