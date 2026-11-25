# Viva guide

Short answers first, then the detail if they ask. Numbers are from `results.js` and
`results/helper_eval.json` as of 2026-11-25; rerun `python -m airpoint.results` before the day.

## The one-minute version

AirPoint is a mouse for people who cannot hold one. A webcam sees the hand; MediaPipe gives 21
points; a small network we trained names the sign; a state machine turns signs over time into
clicks, drags and scrolls. Around that core: a filter that measures a tremor and removes it, a
calibration that fits the system to one person's hand, ways to work with pointing alone or with the
face, signs the user teaches in seconds, speech in, and an AI helper that is supervised step by
step. A JavaScript port runs in a browser from a QR code.

## What is measured, and what is not

| Claim | Evidence | Limit to state |
|---|---|---|
| The network reads signs the rules cannot | 1,508 recorded frames, six signs: rules 50.9% (0% on pinch), network 100% | One person's hand; the 100% is a hold-out of that person's own frames, not a new user |
| Either hand works | The same held-out frames mirrored: 100% after training on mirrored copies | Mirrored data, not a second real hand |
| Fast enough | Tracker about 5 ms a frame, classifier 0.04 ms, 30 frames a second on this laptop | One laptop |
| The tremor filter helps | Simulation: severe tremor, shake left in the cursor 12.0 px with smoothing only, 1.9 px with the canceller; dwell click works 6% then 100% of the time | A simulated tremor. No patient data. |
| The AI helper points accurately and respects its gates | 30 of 30 named buttons, at most 1.6 px off; 5 of 5 errands in a simulated mail app, send and delete both held for a yes | Drawn screens and a simulation, one model, one run |
| The browser version computes the same thing | 3,837 comparisons with Python's answers, all agree | Parity of the maths; the camera path in a browser was checked by hand only |
| Pointing speed (throughput) | Not measured yet | Needs people: see `measurement-day.md` |

## Questions they are likely to ask

**Why not just use the hand-written rules?** We started there. On real recorded frames they are
right half the time, and never on pinch: a pinch seen side-on does not satisfy "thumb near index and
middle finger extended". The network sees all 63 numbers and has no such blind spot. The rules stay
in the app as a fallback (`k`) and as the baseline on the dashboard.

**Why a small network and not a deep one?** The input is already 21 landmarks, not pixels: MediaPipe
did the vision. 63 numbers into 128, 64, 6 trains in seconds on a laptop, runs in 0.04 ms, and can
be retrained for one person in the time it takes to say so. That is what makes calibration and
taught signs possible.

**How do you make it independent of hand size and distance?** Every landmark is taken relative to
the wrist and divided by the palm length (wrist to middle knuckle). A big hand close up and a small
hand far away give the same 63 numbers.

**What is the 1-Euro filter?** A low-pass filter whose cutoff rises with speed: heavy smoothing when
the hand is still, so the cursor does not jitter; light smoothing when it moves, so it does not lag.
Casiez, Roussel and Vogel, CHI 2012.

**Then why another filter for tremor?** A tremor is not jitter: it is a regular oscillation, four to
twelve times a second, and it is fast enough that the 1-Euro filter treats it as intended movement
and lets it through. We measured this before building anything: lowering the cutoff barely helped.
So a meter estimates the tremor's frequency from three seconds of pointer history, and a notch
filter removes just that band. Deliberate movement is slower and passes.

**Is that tested on people with tremor?** No. It is tested on a simulated tremor added to the hand
landmarks, and on synthetic pointing sessions. It is the claim that most needs real users, and the
report says so.

**What does the state machine do that the classifier does not?** The classifier names one frame. The
state machine holds time: a pinch under half a second is a click, a longer one is a drag; a V must
be held 0.3 s before it right-clicks; scrolling repeats while the thumb is held. A five-frame vote
sits between the two so that one misread frame never becomes a click.

**How is this different from what macOS already has?** macOS ships Dwell Control, Head Pointer and
Voice Control. AirPoint differs in: hand signs learned per person, signs taught in seconds, a tremor
that is measured and filtered adaptively, one pipeline across hand and face, an open webcam-only
implementation, and its own measurements. We do not claim to replace those tools.

**What happens to the video?** Nothing is stored and nothing is sent. Frames are read, turned into 21
points and dropped. Recordings for training are landmarks only. The single exception is the AI
helper, which sends a picture of the screen (not the camera) to OpenAI at each step, only after it
has been switched on, with AirPoint's own window blacked out of the picture.

**Is the AI helper safe?** It is supervised, not autonomous. Every step is checked against fixed
limits, shown at the cursor for a second and a half, and stopped by a fist. A step that sends,
pays or deletes waits for a yes, by AirPoint's own word list as well as the model's judgement. At
most twelve steps. The known weakness of all such helpers is text on the screen steering the model;
in our test a message saying "delete everything" was read out as content and nothing was deleted,
but one test is not a proof, which is why the gate does not depend on the model.

**How do you know the browser version is the same system?** By test, not by reading the code. Python
writes a fixture: recorded hands with the features and probabilities it computes, scripts for each
component, and one whole session through the real live loop with every cursor position and click.
The JavaScript must reproduce all of it. 3,837 comparisons, all agree, and the test suite fails if
either side drifts.

**How is it tested?** 331 automated tests that need no camera: every timing rule is a pure class with
the clock passed in, so a test can say "hold a pinch for 0.8 s" exactly. Whole sessions run through
the real loop with synthetic hands and a fake mouse. What tests cannot cover is listed in
`docs/camera-checklist.md`, section by section, for a person to check.

**What would you do next?** Measure with real users, starting with people who have a tremor. Then
calibration that keeps learning during use, and a Windows build (the core is portable; the cursor
glow, speech and sounds are Mac-only today).

## If you do not know

Say what was measured and what was not. Every number on the dashboard has its method written
beside it for that reason.
