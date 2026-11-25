# Slides: outline

Twelve slides, one idea each. Screenshots come from the running app and the dashboard; do not
retype numbers, take them from `results.js` on the day.

| # | Title | On the slide | Say |
|---|---|---|---|
| 1 | AirPoint: a mouse for people who cannot hold one | Name, team, mentor, one picture of a hand driving a cursor | Who it is for, in one sentence |
| 2 | The problem | A mouse needs grip, fine control and a steady hand. Tremor, weakness or a missing hand rule it out. | Existing tools cost money or need special hardware; a webcam is already there |
| 3 | The pipeline | Camera → 21 landmarks → sign → vote → state machine → pointer filter → mouse | Two brains: one reads the shape, one handles time |
| 4 | Reading the hand | 63 numbers: landmarks from the wrist, in palm lengths. Network 63→128→64→6. Table: rules 50.9%, network 100% on 1,508 frames, one person | Why the rules failed on pinch; why the network number is not yet a new-user number |
| 5 | From signs to actions | The state machine diagram. Click under 0.5 s, drag over it, V held 0.3 s, thumbs repeat | Time is what turns a sign into the right action |
| 6 | Tremor | Before and after trace of the cursor. Meter finds the frequency, notch removes it. Simulation table | Why smoothing alone cannot do this. It is a simulation |
| 7 | Fits your body | One-minute calibration: reach, signs you can make, shake. Either hand. Pause | The system adapts to the person, not the reverse |
| 8 | When signs are too much | Dwell and the wheel; face mode; taught signs that speak; talk | One pipeline, four ways in |
| 9 | AI helper | A step shown at the cursor; the yes/no gate. 30/30 pointing, 5/5 errands, drawn screens | Supervised, stoppable, the one part that leaves the laptop |
| 10 | In your browser | The QR code. 3,837 parity checks | Try it now; your picture stays on your phone |
| 11 | What we measured and what we did not | The table from the viva guide | Say the limits before they ask |
| 12 | Next | Real users with tremor; learning during use; Windows | Thank you; questions |

Backup slides: the X-ray view; the test suite (331 tests, no camera needed); the privacy summary;
the comparison with macOS's own accessibility tools.
