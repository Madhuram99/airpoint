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

## Stage 4 — teach a sign, speak for me

- [ ] **4.1 Teach a speaking sign.** `n`, type `hello`, Enter, hold a sign of your own (for example index and little finger up) for 8 s while tilting it, press `1`, type `Thank you for coming`, Enter. After "Learning..." the result line gives a percentage. Back in live view, hold the sign: the laptop says the phrase once, and again only after you open your palm.
- [ ] **4.2 With the mouse off.** Turn mouse control off (`m`) and hold the sign: it should still speak.
- [ ] **4.3 A key sign.** Teach a second sign bound to keys (`2`, then `cmd+c`, or `right` to advance slides). Check it does that and nothing else.
- [ ] **4.4 False fires.** Use AirPoint normally for two minutes with both signs taught. Note every time a taught sign fires when you did not mean it.
- [ ] **4.5 Confusable sign.** Teach a sign very close to a built-in one (a slightly open fist). Does the result line warn you?
- [ ] **4.6 Forget.** `n`, then `1`: the sign is forgotten and no longer fires.
- [ ] **4.7 Voice.** `venv/bin/python -m airpoint.profile madhuram --voice Rishi` (or `Lekha` with a Hindi phrase), relaunch, hold the speaking sign.
- [ ] **4.8 Typing a name with q.** On the name screen type a name containing `q`: AirPoint must not quit.

Report: the percentages from 4.1 and 4.3, the false fires from 4.4, whether 4.5 warned.

## Stage 5 — glow at the cursor, point-only wheel

- [ ] **5.1 Glow.** With the mouse on, a thin ring follows the real cursor. It vanishes when the mouse is off or you enter record / calibrate / teach. `o` switches it.
- [ ] **5.2 What it shows.** Dwell on (`d`): an arc fills at the cursor before the click, then a ripple. Drag: a DRAG tag. Thumb scroll: SCROLL. `p`: PAUSED with the resume countdown. `t`: SIMULATED. A hint (hand out for 3 s) appears under the cursor.
- [ ] **5.3 Click-through.** With a real mouse, click a button that sits under the glow: the click must reach it.
- [ ] **5.4 Above everything.** Put a browser in full screen, switch Spaces, and (if you have one) move to a second display: does the glow stay with the cursor?
- [ ] **5.5 Wheel basics.** `w`, hold the palm still for a second: the wheel opens at the cursor and in the camera view, and the cursor stops following. Stay put: it clicks after a little under a second.
- [ ] **5.6 Wheel items.** Up is talk (Stage 7). Open it again and move up-right (right click), down-right (drag: the cursor follows again, the next stop drops), down (scroll: hand above or below the start point scrolls, resting in the middle ends it), down-left (double click), up-left (pause).
- [ ] **5.7 Timing.** Is there enough time to start moving before the plain click fires? Did any item fire that you did not mean? Was any direction hard to hit?
- [ ] **5.8 Whole computer by pointing.** With no signs at all, open a folder, drag a file, scroll a page and right-click, using only the wheel.
- [ ] **5.9 In the lab.** Take the pointing test with the wheel on: the block should be listed as `airpoint wheel`.

Report: 5.3 and 5.4 exactly as they behaved, and the answers to 5.7.

## Stage 6 — face mode

- [ ] **6.1 Switch.** Mouse on, look at the middle of the screen, press `f`. After a moment the cursor sits near the middle and the banner reads `face`. A box and a nose dot are drawn on your face in the view.
- [ ] **6.2 Head pointing.** Turn your head left, right, up, down: the cursor follows the same way and reaches every screen edge without straining. Is it steady when you hold still?
- [ ] **6.3 Mouth and eyebrows.** Open your mouth briefly: one click. Hold it open and turn your head: a drag; close to drop. Raise your eyebrows for a moment: a right click.
- [ ] **6.4 Wheel with the face.** Hold your head still for a second: the wheel opens. Pick scroll and pause with head movement alone.
- [ ] **6.5 Calibrate the face.** `c`: follow the four prompts (rest, turn, mouth, eyebrows). The result line gives your head range. Does pointing feel better afterwards? Relaunch with `--user madhuram` and press `f`: the calibration is remembered.
- [ ] **6.6 Talking.** Talk normally in face mode and count the unwanted clicks. Then pause (`p`) and talk: there should be none. A still face for two seconds resumes.
- [ ] **6.7 Glasses and light.** If you wear glasses, try with and without; try with a lamp to one side.
- [ ] **6.8 Back to the hand.** `f` again: hand signs work as before and the wheel returns to how you had it.
- [ ] **6.9 In the lab.** Take the pointing test in face mode: the block should be listed as `airpoint face`.

Report: steadiness in 6.2, unwanted clicks in 6.6, and whether 6.5 changed the feel.

## Stage 7 — talk

Once, before these: System Settings › Keyboard › Dictation › on; Shortcut › Customize › press
`control+option+d`. Then `venv/bin/python -m airpoint.speech` should print the same shortcut.

- [ ] **7.1 A key combo with control.** (A fix from this stage: `control` used to be dropped from every combo.) Teach a sign (`n`) with action 2, keys `ctrl+up`. Hold the sign: Mission Control must open.
- [ ] **7.2 Start and stop by wheel.** Mouse on, `w`. Click into a note in Notes with the wheel. Open the wheel and move up (talk): the macOS microphone appears and the banner and the glow read LISTENING with a countdown. Say a sentence: it is typed into the note. Wheel up again: the microphone goes away.
- [ ] **7.3 A sign for it.** `n`, name it, hold a sign, action 6 (talk). Use that sign to start, and again to stop.
- [ ] **7.4 It ends by itself.** Start talk and say nothing for 20 s: the microphone must go away by itself, and AirPoint must read idle. If macOS and AirPoint ever disagree (one listening, one not), note what you did just before; pressing `l` once puts them back in step.
- [ ] **7.5 Mouse off and pause.** While it is listening press `m`: dictation must end first. With the mouse off, talk must refuse to start and say why. The same for `p`.
- [ ] **7.6 Its own window.** Click on the AirPoint window so it is in front, press `l`, speak, press `l`: nothing is typed anywhere, AirPoint must not react to any letter, and it must still be running.
- [ ] **7.7 Face mode.** `f`, start talk from the wheel, speak two sentences: count the clicks while speaking (there should be none). Stop talk: about a second later an open mouth clicks again.
- [ ] **7.8 Hindi.** Switch the dictation language in the macOS microphone pop-up to Hindi and repeat 7.2.

Only if you want the on-laptop engine (it adds spoken commands). These two lines install a package and
download a 148 MB model; AirPoint never does either by itself:

```bash
venv/bin/pip install faster-whisper
venv/bin/python -m airpoint.speech --setup
venv/bin/python -m airpoint.speech --listen 5           # speak: prints what was heard and what it would do
venv/bin/python -m airpoint.profile madhuram --speech local
```

- [ ] **7.9 Microphone.** The first `--listen` asks for microphone access for your terminal. Allow it. The line must show a peak well above 0.01 while you speak; a peak of 0.000 means access was refused.
- [ ] **7.10 Commands and text.** In AirPoint (start-up line reads `talk: on this laptop`): start talk, say "scroll down", stop: the page scrolls and the banner shows `heard "Scroll down."`. Say a full sentence: it is typed. Say "select all", then "copy".
- [ ] **7.11 Silence.** Start and stop talk without speaking: nothing may be typed (the banner says `heard nothing`). Try it once with a fan or music on.
- [ ] **7.12 Speed.** How long from the second tap to the words appearing? Is `tiny.en` (`--model tiny.en`, then `--speech-model tiny.en`) good enough and faster?

Report: 7.1, 7.4 and 7.6 exactly as they behaved; the click count in 7.7; for the on-laptop engine the
peak from 7.9, anything typed in 7.11 and the delay in 7.12.

## Stage 8 — AI helper

This stage sends things off the laptop when you use it: your request and a picture of your main
display at every step, to OpenAI. Close anything you would not want in that picture first.

Before: in the terminal that will start AirPoint, `export OPENAI_API_KEY=...` (your key; never put it
in a file here). Then `venv/bin/python run_demo.py --user madhuram --mouse`.

- [ ] **8.1 Switching on.** `a`: the prompt reads SWITCH ON THE AI HELPER with the red line saying what is sent. Esc: nothing is on. `a`, Enter: the banner now ends with `AI [a] ON`.
- [ ] **8.2 A harmless errand.** Open TextEdit with an empty document and click into it. `a`, type `write hello from AirPoint in this document`, Enter. Watch: THINKING, then the cursor goes to the target, the glow says `AI 1/12: ...` with a countdown, then it acts. At the end a line is spoken. Note the seconds per step.
- [ ] **8.3 A question.** With any page or document in front: `a`, `what is the title of this window?`. The answer is spoken and shown.
- [ ] **8.4 Your own window is hidden.** With the AirPoint camera window visible: `a`, `is there a black rectangle on the screen, and where?`. It should describe one where the AirPoint window is: that is what the model gets instead of your camera picture.
- [ ] **8.5 Stop by fist.** Start a longer errand (`open Calculator and work out 15 times 12`). Make a fist: it stops with the low sound. Keep the fist a moment, then open your hand: nothing may be clicked. A pinch after the open palm clicks as usual.
- [ ] **8.6 Stop by moving.** Start it again and swipe your hand quickly sideways: it stops. Start it once more and rest your hand normally for the whole errand: it must not stop by itself. If it did, say what your hand was doing.
- [ ] **8.7 A step that waits.** In TextEdit with a few words typed: `a`, `close this document without saving`. Before the click that discards it, the helper must stop and ask: the glow says YES OR NO? Thumb down: nothing is lost. Repeat and say yes with a thumb up.
- [ ] **8.8 Yes and no by pointing.** Repeat 8.7 with an open palm only: a two-way wheel appears in the camera view; point up for yes or down for no and hold.
- [ ] **8.9 Wrong clicks.** Over three or four errands in real apps (Finder, Safari, Notes): did it ever click the wrong thing, or miss a small target? Which app?
- [ ] **8.10 One sign, one errand.** `n`, teach a sign with action 7 and a request you would use daily. Hold the sign: the errand starts.
- [ ] **8.11 Face mode.** `f`, then an errand from `a`: eyebrows stop it; on a step that waits, an open mouth says yes.
- [ ] **8.12 Spoken.** Only with the on-laptop speech engine (Stage 7): start talk, say "Computer, open the calculator", stop talk.

Report: seconds per step in 8.2, what it said in 8.4, any false stop in 8.6, whether 8.7 asked before
the discarding click, and every wrong click in 8.9.

## Stage 9 — try it in your browser

```bash
cd ~/Projects/airpoint && venv/bin/python -m http.server 8801      # leave running; open http://127.0.0.1:8801/try/
```

- [ ] **9.1 Starts.** Chrome: the welcome card, Start camera, allow the camera. The status turns green, your mirrored picture appears with the hand skeleton on it. Note the frames a second.
- [ ] **9.2 Signs.** The `sign` line names what you show: palm, pinch, fist, v_sign, thumb up, thumb down. Any it gets wrong, and is it also wrong in the desktop app?
- [ ] **9.3 Tutorial.** Do all six steps with your hand: move, click, double-click, drag, scroll, right-click. Which were hard?
- [ ] **9.4 Pointing test.** Do the test. Is the throughput near what the desktop app gave you in the lab (it has no tremor filter here, and the cursor is inside the page)?
- [ ] **9.5 Dwell.** Switch dwell click on: holding still for a second clicks, with the ring filling.
- [ ] **9.6 Safari and Firefox.** Repeat 9.1 and one tutorial step in each.
- [ ] **9.7 Refusing the camera.** Block the camera: the card must say what to do, not hang.
- [ ] **9.8 The robot.** `http://127.0.0.1:8801/try/?replay&robot=tutorial`: a recorded hand finishes the tutorial by itself (this is what was checked without a camera, and a fallback for the demo).
- [ ] **9.9 Putting it online (your decision).** GitHub › the repo › Settings › Pages › Deploy from a branch › `main` and `/ (root)` › Save. After a minute `https://madhuram99.github.io/airpoint/try/` opens, and `try/qr.png` leads there.
- [ ] **9.10 A phone.** After 9.9: scan the code with a phone, prop the phone up, and try 9.1 to 9.3. Note the frames a second.

Report: frames a second in 9.1 and 9.10, wrong signs in 9.2, anything that failed in 9.6 and 9.7.
