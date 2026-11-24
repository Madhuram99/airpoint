# AirPoint — the one test session

Everything here needs a person at the camera. Stages 1 to 9 were built without one, so this is the
list for the single session at the end. Work top to bottom; later items depend on earlier ones.
For each item note: works / does not work / odd, and anything you had to do differently.

Start AirPoint in one terminal and the lab in another:

```bash
cd ~/Projects/airpoint
venv/bin/python run_demo.py --user madhuram
venv/bin/python -m airpoint.lab
```

## Stage 1 — camera, tremor, X-ray (and the three Week 2 checks)

- [ ] **1.0 Launch line** reads `camera 0 1920x1080 @ 30 fps -> view 640x360`. The banner shows about 30 fps, also in a dim room.
- [ ] **1.1 Feel.** `m` turns the mouse on. Cursor speed and smoothness match Week 1. `z` steps the window size.
- [ ] **1.2 Calibration (Week 2).** `c`: hold still, six poses, "Learning your hand...", then "done" with a percentage. The prompt strip is at the top and never covers the hand. `profiles/madhuram.json` exists; relaunching shows `mlp/madhuram`.
- [ ] **1.3 Dwell (Week 2).** `d`, hold the palm still over an icon: the ring fills in about a second, one click, no repeat until the hand moves away and stops again.
- [ ] **1.4 Mini view (Week 2).** `v`: 320×240 in the top-right corner, staying above a full-screen browser.
- [ ] **1.5 Simulated tremor.** `t`: the cursor shakes and a red line reads SIMULATED TREMOR. Within about 2 seconds it reads LOCKED near 6 Hz and the cursor steadies. `]` makes the shake bigger, `}` faster; the lock follows.
- [ ] **1.6 Filter comparison.** `b` once (smoothing only): a visible wobble returns. `b` again (OFF): the raw shake. `b` again: full.
- [ ] **1.7 A real shake.** Simulator off (`t`). Shake your own hand on purpose for 5 seconds. Does LOCKED appear, and at what frequency?
- [ ] **1.8 False locks.** Two minutes of normal use: move, click, drag, scroll. Note any LOCKED flash.
- [ ] **1.9 X-ray.** `x`: the bars follow your signs, the state boxes light up, the scope and spectrum move, and the frame rate stays near 30.
- [ ] **1.10 Guard rails.** In record (`r`) and calibrate (`c`) the `t` key does nothing.

Report: anything that fails, the frequency from 1.7, any false locks from 1.8.

## Stage 2 — lab and measured numbers

- [ ] **2.1 Lab home.** The browser opens `http://127.0.0.1:8765`. Enter a code (P1), choose AirPoint. With AirPoint running and `m` on, the "AirPoint right now" card shows the same settings as the AirPoint window, and follows when you press `d`, `b` or `t`.
- [ ] **2.2 Tutorial with AirPoint.** All six steps: move, click, double-click (two quick pinches), drag, scroll (keep the cursor over the list), right-click (V-sign). Note any step you had to skip.
- [ ] **2.3 Baseline.** Choose trackpad (or mouse) and take the pointing test once.
- [ ] **2.4 AirPoint, sign click.** Choose AirPoint, dwell off, take the test.
- [ ] **2.5 AirPoint, dwell click.** `d` on, take the test.
- [ ] **2.6 Tremor comparison.** `t` on, then one test each with the filter OFF, smoothing only and full (`b`). These three are the measured version of the simulation table.
- [ ] **2.7 Labels.** "Measured so far" on the lab home lists each condition under the right name.
- [ ] **2.8 Dashboard.** `venv/bin/python -m airpoint.results`, then open `index.html`: the pointing table shows those conditions; nothing reads "not measured yet" that you measured.

Report: the "Measured so far" table as it stands, and any block that landed under the wrong condition.

## Stage 3 — fits your body

- [ ] **3.1 Calibration with reach.** `c`: after HOLD STILL comes REACH (6 s). Move your hand as far as is comfortable in every direction, elbow resting if you like. After "done", the yellow box on the view is your comfort box and small hand movements cover the whole screen.
- [ ] **3.2 Feel in the box.** Does the cursor still feel steady and reach every screen corner? Too jumpy or too slow?
- [ ] **3.3 Other hand.** Use your other hand without recalibrating: the same signs should work.
- [ ] **3.4 Back of the hand.** Turn your hand so its back faces the camera: palm and V-sign should still read.
- [ ] **3.5 A sign you "cannot" make.** Calibrate again but show an open palm when asked for thumb down. The result line should say `skipped: thumb_down`, and thumb down should no longer scroll. Then `venv/bin/python -m airpoint.profile madhuram --reset-roles` and calibrate normally to restore it.
- [ ] **3.6 Pause.** `p`: the banner reads PAUSED and nothing clicks. Hold an open palm still: the ring fills in 2 s and control returns.
- [ ] **3.7 Hints.** Take your hand out for 3 s; cover the camera; hold the hand very close and very far; move it to the frame edge. Each should give a sensible line at the bottom. `h` turns them off.
- [ ] **3.8 Sounds.** Click, right-click, drag and pause each make a sound when the mouse is on. `s` turns them off.
- [ ] **3.9 Role edit by hand.** `venv/bin/python -m airpoint.profile madhuram --role right_click=fist --role grip=pinch`, relaunch: a held fist right-clicks and no longer drags. Restore with `--reset-roles`.

Report: how the comfort box felt (3.2), whether the other hand worked without recalibrating (3.3), any hint that was wrong or annoying.
