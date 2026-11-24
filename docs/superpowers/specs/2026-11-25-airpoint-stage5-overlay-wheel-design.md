# AirPoint Stage 5 — Glow at the cursor, point-only wheel

Date: 2026-11-25
Status: approved in advance (user, 2026-11-25)
Roadmap: `2026-11-25-airpoint-v2-roadmap.md`, Stage 5

## 1. Goal

Two things. Feedback moves from the camera window to where the user is looking: a
glow at the real cursor shows the dwell countdown, clicks, drags, the pause and
hints. And a person who can point but cannot make any sign gets the whole mouse:
holding still opens a wheel of actions around the cursor.

## 2. The wheel — `wheel.py` (pure, any platform)

With wheel mode on (`w`), a one-second dwell opens the wheel instead of clicking.
The system cursor stays where it was; the hand now steers a choice.

Six items, clockwise from the top: click, right click, drag, scroll, double
click, pause.

- Stay near the centre for half a second: plain click. A click costs 1.5 s in all.
- Move toward an item (16 px of hand travel in the 640-wide view) and hold that
  direction for half a second: that item.
- Move far out (70 px) or wait six seconds: the wheel closes and nothing happens.
- **Drag** presses the button and lets the cursor follow again; the next dwell
  drops. **Scroll** freezes the cursor: hand above the start point scrolls up, below
  scrolls down, faster the further; resting in the middle for 1.2 s ends it.
- A sign, a lost hand or leaving live mode closes the wheel and drops a held drag.
- After any action the wheel re-arms only once the hand has moved 30 px, as dwell
  click does.

Choice is by direction, so the wheel works the same under any comfort-box gain.
The wheel is also drawn in the camera view, so it works without the overlay.

## 3. The overlay — `overlay.py` (macOS)

A second process draws a 300-point, borderless, transparent, click-through window
at the screen-saver window level and keeps it centred on the cursor. AirPoint sends
it one JSON line per frame over a pipe: position, ring progress, tag, hint, a
one-shot pulse for an action, and the wheel if open. When the pipe closes the
process exits, so an overlay can never outlive AirPoint. If it fails to start or
dies, AirPoint carries on without it.

It shows: a thin ring at the cursor; the dwell or resume countdown as an arc; an
expanding ripple on click, right click, drag start and drop; a tag (DRAG, SCROLL,
PAUSED, SIMULATED); the current hint; and the wheel with the pointed item lit, its
half-second countdown, and a dot for where the hand is steering.

A spike on this Mac confirmed the window is created at level 1000, ignores the
mouse and is listed by the window server; the drawing code can be rendered
off-screen to an image, which is how its look is checked without a person.

Only shown while mouse control is on. `o` switches it; `--no-overlay` starts
without it.

## 4. Wiring

- `Profile.wheel`; calibration turns it on together with dwell when the hand has no
  clicking sign.
- A new event kind `double_click`.
- The live state for the lab reports wheel mode, so a pointing-test block made with
  the wheel is labelled as such.

## 5. Tests

Wheel: plain click by staying centred; each item by direction; cancel by distance
and by timeout; drag then drop; scroll up, down and exit; a sign closes it and ends
a drag; re-arm distance. Overlay: the client writes one line per state, survives a
dead process; the painter renders every state off-screen (macOS only); the state
built for a click, a pause, an open wheel and mouse-off.

## 6. Limits

The overlay is macOS-only. Whether it stays above a full-screen app on every
Space, and how it behaves on a second display, are for the camera session. The
wheel's distances and times are first values, to be tuned with a person.
