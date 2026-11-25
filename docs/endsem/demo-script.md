# Three-minute demo

One laptop, one terminal, one browser tab with the dashboard. Rehearse it twice with a timer.

## Before the room fills (5 minutes)

```bash
cd ~/Projects/airpoint
venv/bin/python run_demo.py --user madhuram --mouse     # calibrated profile; mouse control already on
open index.html                                         # second screen or behind
```

- Sit where the wall behind you is plain and your face is lit from the front. Check the banner says
  30 fps. If it says 15, the room is too dim: turn a light on.
- Open Notes with an empty note, and a long web page.
- Press `v` once to see the mini view sits in the corner, and again to bring the big view back.
- If you will show the helper: `export OPENAI_API_KEY=...` before starting AirPoint, and check the
  network.

## The three minutes

| Time | Do | Say |
|---|---|---|
| 0:00 | Open palm, move the cursor across the screen. Pinch a folder open. | "A webcam and a hand. No glove, no sensor. Pinch is click." |
| 0:20 | Hold a pinch, drag a window, release. Thumb down on the web page. V sign on a file. | "Hold to drag. Thumbs scroll. A V is right-click. Six signs, read by a small network we trained, thirty times a second." |
| 0:45 | Press `x`. | "This is what it sees: the 21 points, the probability of each sign, the vote, the state machine." |
| 1:00 | Press `t`. The cursor shakes. Wait two seconds for LOCKED. Press `b` twice, then back. | "Now the hard part. This is a simulated tremor, six times a second. Ordinary smoothing cannot hide a regular shake without lagging. So we measure its frequency and cut out exactly that band. Filter off... and on." |
| 1:30 | Press `t` off. Press `w`. Hold still: the wheel opens. Pick scroll. | "Not everyone can pinch. Holding still opens a wheel; you choose by direction. The whole computer by pointing." |
| 1:50 | Press `f`. Move the cursor with your head, open your mouth to click. Press `f`. | "Or no hand at all. Same pipeline, a different tracker in front." |
| 2:10 | Make your taught sign: the laptop speaks. | "And signs you teach it in fifteen seconds. This one speaks for someone who cannot." |
| 2:25 | Switch to the dashboard, Results. | "What we measured, and what we have not. The rules we started with read half the frames right; the network reads them all, on one person's hand. Tremor numbers are a simulation and say so." |
| 2:45 | Show the QR code. | "It also runs in your browser. Scan it and try with your own hand; your camera picture never leaves your phone." |

## If something fails

| What | Do |
|---|---|
| The hand is not tracked | Step back to half an arm, palm facing the camera, light on your hand. `h` shows hints. |
| The cursor is jumpy | `c` and recalibrate (one minute), or `b` to check the filter is on `full`. |
| A sign will not fire | `k` switches to the hand-written rules; or use dwell (`d`) for clicks. |
| The camera will not open | Quit other apps using it. `venv/bin/python run_demo.py` again. |
| Everything fails | The backup video, or `http://127.0.0.1:8801/try/?replay&robot=tutorial` after `venv/bin/python -m http.server 8801`: a recorded hand doing the tutorial in the browser. |
| The helper is slow or the network is down | Skip it. It is three to four seconds a step when it works. |

## Do not

- Do not demo the AI helper on a screen showing anything private: it sends a picture of the screen.
- Do not claim accuracy on new users, or pointing throughput, unless measurement day has happened.
- Do not say the tremor filter was tested on patients. It was not.
