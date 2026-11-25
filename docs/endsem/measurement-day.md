# Measurement day

The one thing no code can do: other people's hands and other people's pointing. An hour with the
team gives the report its two missing numbers. Until then the dashboard says "not measured yet" in
those places, and it should.

## What is missing, and why it matters

| Number | Now | After this session |
|---|---|---|
| Pose accuracy on a hand the model never saw | not known: one person recorded, so 100% is a hold-out of that person's own frames | leave-one-person-out over 4 people |
| Pointing throughput (ISO 9241-411) | nothing recorded | bits per second for mouse, trackpad and each AirPoint mode, per person |

## 1. Record each person (5 minutes each)

```bash
venv/bin/python run_demo.py --record --user <first name>
```

Space records all six signs in turn, five seconds each. Ask for variety while holding: tilt, closer,
further. `1` to `6` redo one sign. Do both hands if there is time (a second run adds rows).
Check the `kept` count: about 150 frames a sign.

## 2. Retrain and read the honest number

```bash
venv/bin/python -m airpoint.train          # prints leave-one-user-out accuracy, per person and per sign
venv/bin/python -m airpoint.export_web     # the browser demo gets the new network
node try/parity.mjs                        # and must still agree with Python
```

With two or more people the report line changes by itself from "hold-out of one user" to
"leave-one-user-out over N users". Write down the per-person figures: the lowest one is the
interesting one.

## 3. Pointing test (10 minutes each)

Two terminals:

```bash
venv/bin/python run_demo.py --user <name> --mouse
venv/bin/python -m airpoint.lab
```

In the lab page enter the person's code, then one block each, in a different order per person:

1. mouse
2. trackpad
3. AirPoint, sign click (calibrate first with `c`, about a minute)
4. AirPoint, dwell click (`d`)
5. AirPoint, wheel (`w`), if time
6. AirPoint, face (`f`), if time

Do the tutorial once before the first AirPoint block; it is practice, not data. A block is 27
selections and takes about a minute. The page labels each block from what AirPoint is really doing,
so nobody has to type the condition.

For the tremor claim, one extra pair per person: AirPoint with the simulated tremor on (`t`), once
with the filter on `full` and once on `smooth only` (`b`). This is still a simulated tremor, on a
real person's pointing. Say so in the report.

## 4. Put it on the dashboard

```bash
venv/bin/python -m airpoint.results        # rebuilds results.js from results/trials.jsonl and the new model
open index.html
git add data models try results results.js && git commit -m "data: measurement day" && git push
```

## Things to say out loud before starting

- Each person is recorded as hand landmarks only: 21 points a frame. No picture is saved.
- Their first name goes into a public repository with those points. Ask, and use a code instead if
  anyone prefers.
- Four people is a pilot, not a study. The report must call it that.
