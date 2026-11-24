# AirPoint Stage 6 — Face mode

Date: 2026-11-25
Status: approved in advance (user, 2026-11-25); model download approved in chat the same day
Roadmap: `2026-11-25-airpoint-v2-roadmap.md`, Stage 6

## 1. Goal

A person with no usable hand drives the same AirPoint with their head: turning the
head moves the cursor, opening the mouth clicks or drags, raising the eyebrows
right-clicks, and the point-only wheel supplies everything else. Nothing after the
tracker changes: the same vote, state machine, filters, dwell, wheel, effects, lab
and glow. One brain, another body part.

## 2. Measured first (this laptop's camera, a real face, 2026-11-25)

- MediaPipe FaceLandmarker (`face_landmarker.task`, 3.76 MB, from the official model
  bucket) returns 478 landmarks, 52 expression scores named `jawOpen`,
  `browInnerUp`, … and a head-pose matrix, in 5.4 ms per frame at 30 fps.
- Head direction from the pose matrix moves with the nose's position inside the
  face (correlation +0.97 for left-right), which fixes its sign: in the mirrored
  view, a larger yaw is a head turned toward the right of the picture. Looking up
  is a larger pitch.
- At rest the expression scores `jawOpen` and `browInnerUp` read 0.00.

## 3. Design

- **`face.py`.** `FaceTracker.detect(frame)` returns yaw and pitch in degrees, the
  expression scores and a few landmarks for drawing, or nothing. Without the model
  file it raises with the one-line download command, as the hand tracker does.
- **Pointer.** `FacePointer` maps the comfortable head range onto the whole view:
  x = W·(0.5 + (yaw − yaw₀) / 2·span), y likewise with pitch inverted. Defaults:
  ±18° across, ±12° up and down, around the resting pose. The result goes through
  the usual pointer pipeline, with the whole frame as the active region.
- **Signs.** `FaceSigns` turns scores into the labels `neutral`, `mouth_open`,
  `brows_up` with hysteresis (on at the threshold, off at 60% of it), and into the
  probabilities the X-ray panel draws. Roles: `mouth_open` grips, `brows_up` held
  0.3 s right-clicks.
- **Calibration (`c` in face mode).** Rest facing the screen 3 s; turn the head as
  far as is comfortable 6 s; open the mouth 3 s; raise the eyebrows 3 s. That sets
  the resting pose, the spans (the smaller side of each range, so both screen edges
  stay reachable), and each threshold halfway between the resting noise and the
  expression's peak. An expression that never rises 0.15 above rest is dropped.
  Saved in the profile; no training involved.
- **Switching.** `f` toggles hand and face in live mode. Entering face mode turns
  the wheel on, because the wheel is how a face scrolls, drags and pauses; leaving
  restores what was set. Record, teach and the tremor simulator are hand-only and
  refuse in face mode.

## 4. Risks stated plainly

- Talking opens the mouth, which clicks. Pause (`p`, or the wheel's pause item)
  before talking; a still face unpauses exactly as a still palm does.
- Glasses, beards and side lighting change the expression scores; calibration sets
  the thresholds per person, and the X-ray panel shows the live scores.
- Holding the head turned is tiring; the spans come from the comfortable range, and
  the cursor rests at screen centre when the head is at rest.
- macOS ships its own head pointer. The point of this one is the shared pipeline:
  the same tremor canceller, wheel, lab and profile serve a hand and a face.

## 5. Tests

Pose angles from known rotations; pointer mapping, centre and clamps; sign
hysteresis and probabilities; calibration results from synthetic recordings,
including a dropped expression; a session driven by synthetic face observations:
head turn moves the cursor, a short mouth-open clicks, a long one drags, raised
brows right-click, `f` restores the hand roles; the tracker on a blank frame.
