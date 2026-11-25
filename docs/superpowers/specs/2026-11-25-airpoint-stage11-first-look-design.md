# Stage 11: fixes from the first look

Date: 2026-11-25. Approved by the user in chat ("go") after their first session with the app.

## What the first look found

1. The window is a debug read-out: jargon (`RESTING | pose none 0.00 | ratio 1.00 | mlp`), blurred text
   (drawn at 640 wide, then stretched), and nothing that says what a key or a sign does.
2. The glow at the cursor is a plain ring that means nothing to someone seeing it for the first time.
3. The wheel shows nothing when `w` is pressed (it opens only after a still hand), so it looks broken.
4. Talk did not work: `l` needs mouse control on, the default engine presses a dictation shortcut that
   was never set up, and words dictated with `l` are lost because AirPoint's own window is in front.
5. With the on-laptop speech engine the whole app froze the first time `l` was used with a real
   microphone. The frame loop stopped and the app could only be killed.

## The freeze

Not reproduced without the real microphone. Ruled out, each with a run of the real app loop or a probe:
the recognizer on a thread (fake microphone), a CoreAudio stream with a Python callback (silent output
stream), 480 forks during live audio callbacks, key events made on a worker thread, a mouse press held on
the app's own title bar, and the app's own menu held open. What is left is the microphone input stream
living inside the app process.

Decision: the app process no longer touches the microphone or the recognizer. Both move to a second
process (`python -m airpoint.speech --serve`), the way the cursor glow already works:

- the app writes `start` / `stop` / `quit` lines; the process answers with JSON lines
  (`ready`, `listening`, `heard`, `problem`);
- the microphone is read with blocking reads on that process's main thread: no audio callback runs
  Python code, so no audio thread ever waits for the interpreter;
- what was said is recognised and sent back before the stream is closed, and a watchdog ends the process
  if closing takes more than two seconds, so a stuck audio call costs one restart, never the words;
- the app gives every step a time limit and replaces a process that misses it;
- AirPoint dying closes the pipe, which ends the process and with it the microphone.

Sounds and other helper programs are started with `posix_spawn` instead of `fork`, so the frame loop
never forks a process that has live audio and tracker threads.

## Talk from the keyboard

AirPoint remembers the last app that was in front before its own window. When words arrive while
AirPoint's window is in front, that app is brought forward and the words are typed there. With no such
app the words are not typed and the message line says so.

## The window

Camera on the left, a side panel on the right, one message line underneath. Nothing covers the camera
picture in the live view.

```
+--------------------------------------------+--------------------------+
|                                            |  Moving                  |
|                                            |  Open palm moves the     |
|              camera picture                |  cursor                  |
|      (hand, comfort box, pointer)          |                          |
|                                            |  [hand] palm    move     |
|                                            |  [hand] pinch   click... |
|                                            |  ... six signs ...       |
|                                            |                          |
|                                            |  [M] Mouse  on   [D] ... |
|                                            |  ... the switches ...    |
+--------------------------------------------+--------------------------+
|  what the key you just pressed did, a hint, or what was heard         |
+-----------------------------------------------------------------------+
```

- Side panel, top: one plain word for what is happening (Resting, Moving, Pressing, Dragging, Scrolling,
  Paused) and one line saying what the sign in view does.
- The signs: one row per sign with a small drawing of the hand (taken from a real recorded hand), its
  name and what it does under this user's profile. The row of the sign in view is lit. Face mode lists
  the face's signs instead. Taught signs follow the six.
- The switches: key, name, state. Mouse, dwell, wheel, glow, filter, face, talk, AI helper, pause.
- Message line: what the last key did, in a sentence; otherwise the camera hint, what was heard, what the
  AI helper is doing, or the tremor read-out.
- Numbers (confidence, pinch ratio, classifier, frame rate) live in X-ray (`x`), which takes the side
  panel's place while it is on.
- Sharp text: the picture is drawn at the display's real pixel size (zoom x backing scale) and the window
  is sized in points. Text is the system font where it exists, OpenCV's built-in sans elsewhere.

Colour carries two meanings only. Blue: AirPoint is tracking. Yellow: an action is happening or a choice
is lit. Red is kept for warnings. Each colour is always paired with a word or a shape.

| name | value | use |
|---|---|---|
| ink | `#14181F` | panel ground, text on yellow |
| paper | `#F4F6F8` | text on ink |
| mist | `#8C96A5` | things that are off |
| access blue | `#3D7BFF` | tracking: skeleton, pointer, comfort box, glow |
| signal yellow | `#FFD23F` | actions and the lit row |
| alert red | `#FF5A52` | warnings |

## The glow

Blue ring with a soft halo while the cursor follows. Yellow while a button is pressed, dragging or
scrolling. A word under it for a moment after each action (click, right click, double click) and for as
long as a state lasts (dragging, scroll up, paused, listening). The dwell countdown stays an arc.

## The wheel

Larger, an icon and a word in each slice, the pointed slice filled yellow, the centre says click. One line
under it says how to use it. Pressing `w` shows the wheel at the cursor for a moment with the line "hold
still for 1 s to open it", so the key visibly does something.

## Not in this stage

Clicking the panel with the hand, a settings screen, themes, and a root-cause proof of the freeze (it
needs the real microphone and the user's yes).
