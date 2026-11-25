# AirPoint: end-semester report (draft)

Status: a draft in plain text. The department's format, the front matter and the performa are
still to be supplied; two results need the measurement day and are marked **[to measure]**.
References are listed to be checked against the sources before submission.

## Abstract

AirPoint lets a person control a computer with hand signs seen by an ordinary webcam. It is built
for people who cannot use a mouse: a hand that shakes, cannot grip, or is not there. MediaPipe
supplies 21 hand landmarks per frame; a small neural network trained on those landmarks names the
sign; a state machine turns signs over time into clicks, drags, scrolls and right-clicks. Around
this core the system measures and cancels tremor, calibrates itself to one person's reach and
ability, works by pointing alone or by face, accepts signs taught by the user in seconds, takes
speech, and offers a supervised AI helper. A JavaScript port, verified against the Python code by
3,837 automated comparisons, runs in a browser. On 1,508 recorded frames from one person the
network reads every held-out frame correctly where hand-written rules read 50.9%. In simulation the
tremor canceller reduces the shake left in the cursor under a severe tremor from 12.0 px to 1.9 px.
Accuracy on unseen users and pointing throughput are **[to measure]**.

## 1. Problem

A mouse assumes a hand that can grip, move finely and hold still. Essential tremor, Parkinson's
disease, muscular weakness, arthritis and limb difference each break that assumption. Assistive
devices exist (head pointers, eye trackers, switch systems) but need hardware, and the built-in
alternatives are fixed in what they recognise. A webcam is already on every laptop. The question
this project asks is how far a webcam alone can go when the software adapts to the person.

## 2. Related work (verify each before submission)

- MediaPipe Hands: real-time hand landmarks on device (Zhang et al., 2020).
- The 1-Euro filter: speed-adaptive low-pass filtering for pointing (Casiez, Roussel, Vogel, CHI 2012).
- ISO 9241-411 and the effective-width method for pointing throughput (Soukoreff and MacKenzie, 2004).
- Pointing aids for motor impairment: Steady Clicks (Trewin et al., 2006), the Angle Mouse (Wobbrock et al., 2009).
- Face-gesture pointing on MediaPipe: Google Project Gameface (2023).
- Voice with pointing: "Put-That-There" (Bolt, 1980).
- Platform tools: macOS Dwell Control, Head Pointer and Voice Control.

## 3. Design

### 3.1 Pipeline

Camera (mirrored, 30 fps) → MediaPipe hand landmarks → features → classifier → five-frame vote →
state machine → pointer filters → comfort box → operating system. Everything after the tracker works
in a 640-pixel-wide canonical view, so thresholds do not depend on the camera's resolution.

### 3.2 Reading the sign

Each frame's 21 landmarks are shifted so the wrist is the origin and divided by the palm length,
giving 63 numbers that do not change with hand size or distance. A multilayer perceptron
(63→128→64→6, ReLU) is trained on recorded frames with tilted, slightly noisy and mirrored copies. A
majority vote over five frames, with a confidence floor of 0.75, keeps single misread frames from
acting.

### 3.3 From signs to actions

A finite state machine with the clock as an input: resting, moving, pressing, dragging, holding,
scrolling. A grip released within 0.5 s is a click; held longer it is a drag. A V sign held 0.3 s
right-clicks once and re-arms on a neutral pose. Thumbs scroll while held. Which sign plays which
role is stored per user.

### 3.4 The pointer and tremor

The pointer passes through a 1-Euro filter, smoothed harder while the fingers close for a pinch so
the click lands where it was aimed. A simulation done before any code showed that this cannot
remove a periodic tremor: the oscillation is fast enough to open the filter. A tremor meter
therefore estimates the dominant frequency in the 3 to 12 Hz band from three seconds of pointer
history, locks when successive estimates agree, and drives a notch filter at that frequency ahead
of the smoothing.

### 3.5 Fitting the person

A one-minute calibration measures resting shake, the comfortable reach (which becomes the box that
maps to the whole screen), and which signs the hand can hold; signs it cannot hold are dropped and
dwell click takes over if no clicking sign is left. A personal model is retrained with the user's
own frames and swapped in while the app runs.

### 3.6 Other ways in

Dwell click, and a wheel that opens at the cursor and is steered by direction, serve a hand that
can only point. Face mode replaces the hand tracker with a face tracker: head pose moves the
pointer, mouth and eyebrows act. The user can teach a new sign in fifteen seconds and bind it to a
spoken phrase, keys, text, an app, dictation or an errand.

### 3.7 AI helper

One request, carried out one step at a time by a vision-language model that sees a picture of the
screen. Each proposed step is validated against fixed limits, shown at the cursor, and can be
stopped by a fist; steps that cannot be undone wait for an explicit yes, decided by the system's
own rule as well as the model's. The helper is off by default and is the only component that sends
data off the machine.

### 3.8 In the browser

The features, network, vote, pointer filter, state machine, dwell and test statistics are ported to
JavaScript and driven by MediaPipe's browser build. A fixture generated by the Python code holds
the port to it.

## 4. Implementation

Python 3.12, MediaPipe, scikit-learn, OpenCV; about 5,700 lines in the package and 1,100 lines of
JavaScript. Timing logic lives in pure classes with the clock passed in, which is what makes it
testable without a camera: 331 automated tests, including whole sessions through the live loop
with synthetic hands and a recording mouse.

## 5. Evaluation

| What | Method | Result |
|---|---|---|
| Sign recognition | 1,508 frames, one person, 25% hold-out | network 100%; hand-written rules 50.9% (pinch 0%, thumb down 7%, thumb up 26%, fist 83%, V 88%, palm 100%) |
| Other hand | the held-out frames mirrored | 100% |
| Unseen users | leave-one-person-out | **[to measure]** |
| Speed | this laptop | tracker ≈ 5 ms, classifier 0.04 ms per frame; 30 fps |
| Tremor, moderate (14 px, 6 Hz) | simulation, 3 sessions of 60 s | shake left 4.2 → 1.0 px; dwell works 96% → 100% |
| Tremor, severe (25 px, 5 Hz) | same | 12.0 → 1.9 px; 6% → 100% |
| Tremor, very severe (40 px, 6 Hz) | same | 20.6 → 3.3 px; 0% → 100% |
| Pointing throughput | ISO 9241-411, effective width | **[to measure]** |
| AI helper, pointing | 30 named buttons on drawn screens | 30 hits; at most 1.6 px from centre; 3.6 s per step |
| AI helper, errands | 5 in a simulated mail app | 5 passed; send and delete held for a yes; injected instruction ignored |
| Browser port | fixture against Python | 3,837 of 3,837 comparisons agree |

## 6. Limitations

- One person's recordings. The recognition figure is not a generalisation figure.
- No participant with a motor impairment has used the system. The tremor results are a simulation.
- Pointing performance is unmeasured.
- The AI helper was evaluated on drawn screens and a simulation, not real applications; such
  helpers can be steered by on-screen text, and the confirmation gate is a mitigation, not a cure.
- macOS only for the cursor overlay, speech output and sounds. One camera, one hand at a time.
- Parts that need a person at the camera (listed in `docs/camera-checklist.md`) were verified by
  tests and simulation, and by a first-hand session **[to complete]**.

## 7. Future work

A study with users who have tremor; calibration that continues during use; a Windows build; a
second hand for modifiers.

## 8. Team and declaration

**[group number, session, members and branches, mentors, AI-use declaration: from the performa]**
