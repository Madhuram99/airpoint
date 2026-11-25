# AirPoint Stage 9 — Try it in your browser

Date: 2026-11-25
Status: approved in advance (user, 2026-11-25); built the same day. Published with GitHub Pages later that
day, on the user's go: https://madhuram99.github.io/airpoint/try/
Roadmap: `2026-11-25-airpoint-v2-roadmap.md`, Stage 9

## 1. Goal

Someone in the audience scans a QR code and controls a cursor with their own hand, in their own
browser, with nothing to install: the tutorial and the pointing test from Stage 2, with their own
score at the end. The video never leaves their device.

## 2. Design

A static folder, `try/`, that works from any plain web server (so also from GitHub Pages):

- `airpoint.js`: a port of the parts of the Python package the demo needs, with no browser code in
  it: hand features, the pose network's forward pass, the vote, the 1-Euro pointer filter, the state
  machine, dwell click, the mapping to the screen, and the pointing-test statistics. One `Pipeline`
  class ties them together the way the live loop does.
- `model.json`: the trained pose network (63 → 128 → 64 → 6) written out as numbers by
  `python -m airpoint.export_web`.
- `parity.json`: inputs and the answers Python gives for them, written by the same command: real
  recorded hands with their features and probabilities; a vote sequence; a pointer trajectory; a
  state-machine script; dwell; a scored test block; and one whole session run through the real
  Python `Session`, with every cursor position and mouse action it produced.
- `parity.mjs`: runs the port on those inputs and compares. `node try/parity.mjs` in a terminal, the
  same check inside the page (`?selftest`), and a pytest that runs the Node check.
- `pointer.js`: a virtual pointer. It turns the pipeline's cursor and events into ordinary mouse
  events inside the practice frame (move, enter and leave, down, up, click, double click, context
  menu, scrolling), so the Stage 2 pages run unchanged.
- `app.js`, `index.html`: camera, MediaPipe hand tracking (the browser build, same version as the
  Python one, loaded from a public CDN; the hand model is the file already in this repo), the
  mirrored preview with the skeleton, and the practice pages in a frame beside it.
- `lab/lab.js` gains one hook: embedded in this page there is no server, so the page asks its
  parent to score the test, with the ported statistics. Nothing is stored or sent.

Not ported: the tremor canceller, calibration, taught signs, the wheel, face mode, talk and the
helper. The page says it is a taste of the real thing.

## 3. Checking it without a camera

- Parity, automatically: every ported function against Python, to 1e-9 where the arithmetic is the
  same, and the whole-session script event for event.
- In a real browser, driven by a replay: the page accepts recorded hands instead of a camera
  (`?replay`), so the virtual pointer, the practice pages and the scoring can be exercised end to
  end. The hand-tracking engine is loaded and run on an empty picture to show it starts.

Done on 2026-11-25, in Chrome on this laptop, from a local web server:
- parity: 3,837 checks, all agree, in Node and inside the page;
- the hand-tracking library (1.2.3, from the CDN) and the hand model load, and find no hand in an
  empty picture;
- a robot hand (`robot.js`), feeding recorded hands in at 30 a second and steering by looking at the
  page, finished all six tutorial steps in 20 s and the whole pointing test (30 clicks, none
  missed); the result card was filled in by the ported statistics. No errors in the console.

## 4. Not checked without a person

The camera itself: permission, a real hand in a real room, phones.
