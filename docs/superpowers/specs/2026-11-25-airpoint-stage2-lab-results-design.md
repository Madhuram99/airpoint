# AirPoint Stage 2 — Practice pages and measured numbers

Date: 2026-11-25
Status: approved in advance (user, 2026-11-25: "do stage 2 to 10 in sequence, it's a go from me in all")
Roadmap: `2026-11-25-airpoint-v2-roadmap.md`, Stage 2

## 1. Goal

Give the project a measuring instrument and make the dashboard honest. A local web
app teaches a new user the gestures and runs the standard pointing test. Python
turns the raw trials into throughput and error rate. The dashboard shows only
numbers that were measured or are clearly labelled as simulation; where nothing has
been measured yet it says so.

## 2. Why now

Found while reading the dashboard:
- Its Fitts throughput table comes from movement times typed into the benchmark
  script ("configured pilot values"), shown as a result with "+83.5% vs Raw".
- The HTML carries invented fallback numbers (per-gesture accuracy, F1) that show
  whenever `results.js` is missing, and `results.js` is ignored by git.
- The headline "99.9% accuracy" is measured on augmented copies of 20 seed rows,
  which the 13 November spec already called "not a generalization number".

Every later stage (comfort box, dwell wheel, face mode) needs a fair way to be
compared. This stage builds it once.

## 3. What the user gets

```
venv/bin/python -m airpoint.lab        # serves http://127.0.0.1:8765 and opens it
venv/bin/python -m airpoint.results    # rebuilds results.js for the dashboard
```

- **Lab home**: participant code, input device, the live AirPoint settings it
  detected, links to the two pages, and the sessions saved so far.
- **Tutorial**: six short steps (move, click, double-click, drag, scroll,
  right-click), each naming the AirPoint sign for it. Doubles as the safe demo
  sandbox. Time per step is saved.
- **Test**: the ISO 9241-411 multi-directional tapping task. Nine targets on a
  ring, three difficulty levels, 27 timed selections, about two minutes. The score
  appears at the end.
- **Dashboard**: accuracy on the real recorded hands, measured pointing results per
  condition, the tremor simulation table from Stage 1, measured latency.

## 4. Measurement method

Task: N = 9 circular targets of width W on a ring of diameter D. The first click
(on the top target) starts a sequence; each selection then crosses the ring to the
target ((N+1)/2 steps on), so every movement is close to D long, and the ninth
returns to the start. Every click counts and moves on; a click outside the target
is an error, not a retry. Three sequences per block, in random order:

| D (px) | W (px) | ID = log2(D/W + 1) |
|---|---|---|
| 320 | 96 | 2.12 |
| 480 | 64 | 3.09 |
| 640 | 48 | 3.84 |

If the window is too small the whole layout scales and the real pixel sizes are
recorded; the maths uses what was shown.

The page records raw events only: target centres, click point, timestamps
(`performance.now`), hit or miss, and the pointer path. Python computes, per
sequence, following Soukoreff & MacKenzie (2004):

- task axis `u` = unit vector from the previous target's centre to this one;
- `dx` = (click − target centre) · u, the over- or undershoot along the axis;
- `We = 4.133 × SD(dx)` (sample SD), the effective width;
- `Ae` = mean of (click − previous click) · u, the effective distance;
- `IDe = log2(Ae / We + 1)`, `MT` = mean time between selections;
- throughput `TP = IDe / MT`, in bits per second.

A block's throughput is the mean over its sequences; a condition's is the mean over
participants. Error rate is misses over selections. Two path measures come free
from the recorded path and matter for a shaking hand: path length over straight
distance, and target re-entries.

Limits to state: a dwell click's movement time includes the one-second dwell; a
sequence whose clicks all land on one line has no spread, and then the nominal W is
used and the sequence is flagged.

## 5. Knowing the condition

A wrong label ruins a comparison, so AirPoint tells the lab what it is doing. While
running, the session writes `results/live.json` once a second: user, mode, mouse
on, dwell on, filter mode, simulator on with size and speed, tremor lock, classifier,
frame rate. The lab server reads it (stale after 3 s = not running), the page shows
it, and the block is saved with the snapshot from its start and its end. Mouse and
trackpad baselines are chosen by hand.

## 6. Parts

```
airpoint/lab/stats.py    sequence, block and condition statistics (pure)
airpoint/lab/store.py    results/trials.jsonl: append, read
airpoint/lab/server.py   standard-library HTTP server: static pages, /api/state,
                         /api/blocks (save + score), /api/summary
airpoint/lab/__main__.py `python -m airpoint.lab [--port N] [--no-open]`
lab/                     index.html, test.html, tutorial.html, lab.css, lab.js,
                         test.js, tutorial.js  (plain files, no build step, no CDN)
airpoint/sim_pointer.py  the synthetic pointer sessions (moved out of tests/)
airpoint/results.py      `python -m airpoint.results` → results.js
airpoint/session.py      writes results/live.json
index.html               reads everything from results.js; no invented fallbacks
legacy/                  the midterm benchmark script, kept, writing its own file
```

`results.js` becomes a tracked file so the dashboard works from a fresh clone and,
in Stage 9, when hosted. It holds:
- dataset: rows, users, rows per pose;
- classifier: per-pose precision and recall from the training report with its
  method named honestly (leave-one-user-out over N users, or a row hold-out of one
  user), and the rule classifier's accuracy on the same real frames;
- latency: classifier time per frame measured on real rows;
- tremor: the Stage 1 simulation table (smoothing only vs canceller), labelled
  simulation;
- pointing: per condition, participants, selections, throughput, error rate, mean
  time; empty until someone runs the test.

The server binds to 127.0.0.1 only, serves a fixed set of folders, and caps the
size of a saved block.

## 7. Honesty rules

- No number on the dashboard without a source in `results.js`.
- Nothing measured yet reads "not measured yet", never a default.
- Simulation is labelled simulation. A one-user hold-out is labelled as such.
- The midterm benchmark script is kept under `legacy/` and can no longer overwrite
  the dashboard's data.

## 8. Tests

- Statistics against a hand-computed sequence; error rate; the flat-spread fallback;
  path ratio and re-entries; block and condition aggregation.
- Server: static files, path traversal refused, state fresh and stale, a block is
  saved and scored, oversized bodies refused.
- Session writes the live state at most once a second.
- Results generator on a temporary project: dataset, classifier, tremor table,
  pointing summary, empty state.
- Pages: `node --check` on every script; one browser run of the tutorial and the
  test page against the real server.

## 9. Not in this stage

Human numbers. The user runs the test for pilot numbers in the single test session;
teammates on measurement day (Stage 10). The browser-only version of these pages is
Stage 9. Gesture cards on the dashboard still say five gestures; the final
dashboard pass is Stage 10.
