// AirPoint in the browser: a port of the parts of the Python package the demo needs.
// No browser code in here. Every function is checked against Python's own answers by parity.mjs.
// Times are seconds; pointer positions are pixels of the 640-wide canonical view, as in Python.

export const WRIST = 0, THUMB_TIP = 4, INDEX_TIP = 8, MIDDLE_MCP = 9;
export const NONE = "none", UNKNOWN = "unknown";
export const GRIP = new Set(["pinch", "fist"]);
export const SCROLL = { thumb_up: 1, thumb_down: -1 };
export const HOLD_EVENTS = { v_sign: "right_click" };
export const FC_MIN = 0.80, FC_MIN_PRECLICK = 0.30, BETA = 0.030, PRECLICK_RATIO = 0.42;
export const ACTIVE_REGION = [0.15, 0.15, 0.85, 0.80];
export const CANON_W = 640;
const PRESS_HOLD_S = 0.50, HOLD_FIRE_S = 0.30, SCROLL_PERIOD_S = 0.10, SCROLL_TICKS = 2;
const DWELL_S = 1.0, DWELL_STILL_PX = 15.0, DWELL_REARM_PX = 30.0;

// ---- features (features.py)

/** 21 [x, y, z] landmarks -> 63 numbers: wrist at the origin, divided by the palm length. */
export function normalize63(pts) {
  const w = pts[WRIST];
  const rel = pts.map(p => [p[0] - w[0], p[1] - w[1], p[2] - w[2]]);
  const m = rel[MIDDLE_MCP];
  const palm = Math.sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]) + 1e-6;
  const out = new Float64Array(63);
  for (let i = 0; i < 21; i++) for (let a = 0; a < 3; a++) out[i * 3 + a] = rel[i][a] / palm;
  return out;
}

const d2 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

/** Thumb tip to index tip over the palm length, in the picture plane. */
export function pinchRatio(pts) {
  return d2(pts[THUMB_TIP], pts[INDEX_TIP]) / (d2(pts[MIDDLE_MCP], pts[WRIST]) + 1e-6);
}

/** How big the hand looks: wrist to middle knuckle in pixels of a w x h view. */
export function palmPx(pts, w, h) {
  return Math.hypot((pts[MIDDLE_MCP][0] - pts[WRIST][0]) * w, (pts[MIDDLE_MCP][1] - pts[WRIST][1]) * h);
}

// ---- the pose network (classifier.py): relu layers, softmax out

export class Mlp {
  constructor(model) {
    this.labels = model.labels;
    this.layers = model.layers.map(l => ({ w: l.w.map(row => Float64Array.from(row)), b: Float64Array.from(l.b) }));
  }

  forward(x) {
    let a = x;
    this.layers.forEach((layer, li) => {
      const out = Float64Array.from(layer.b);
      for (let i = 0; i < a.length; i++) {
        const ai = a[i], row = layer.w[i];
        if (ai !== 0) for (let j = 0; j < out.length; j++) out[j] += ai * row[j];
      }
      if (li < this.layers.length - 1) for (let j = 0; j < out.length; j++) if (out[j] < 0) out[j] = 0;
      a = out;
    });
    let max = -Infinity, sum = 0;
    for (const v of a) if (v > max) max = v;
    const p = a.map(v => Math.exp(v - max));
    for (const v of p) sum += v;
    return p.map(v => v / sum);
  }

  proba(pts) { return this.forward(normalize63(pts)); }

  /** -> [label, confidence] */
  predict(pts) {
    const p = this.proba(pts);
    let best = 0;
    for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
    return [this.labels[best], p[best]];
  }
}

// ---- the vote (classifier.py)

/** Majority label over the last 5 frames; a strict majority with mean confidence >= 0.75, else unknown. */
export class VoteSmoother {
  constructor(window = 5, minConf = 0.75) { this.window = window; this.minConf = minConf; this.buf = []; }
  reset() { this.buf = []; }
  update(label, conf) {
    if (label === null || label === undefined) { this.buf = []; return NONE; }
    this.buf.push([label, conf]);
    if (this.buf.length > this.window) this.buf.shift();
    const counts = new Map();
    for (const [l] of this.buf) counts.set(l, (counts.get(l) || 0) + 1);
    let winner = null, n = 0;
    for (const [l, c] of counts) if (c > n) { winner = l; n = c; }
    if (n * 2 <= this.buf.length) return UNKNOWN;
    let sum = 0;
    for (const [l, c] of this.buf) if (l === winner) sum += c;
    return sum / n >= this.minConf ? winner : UNKNOWN;
  }
}

// ---- the pointer (filters.py, pointer.py in its "smooth" mode)

/** Casiez, Roussel, Vogel (CHI 2012): cutoff = fc_min + beta * |speed|. */
export class OneEuroFilter {
  constructor(fcMin = FC_MIN, beta = BETA, dCutoff = 1.0) {
    this.fcMin = fcMin; this.beta = beta; this.dCutoff = dCutoff;
    this.xPrev = null; this.dxPrev = 0; this.tPrev = null;
  }
  static alpha(fc, dt) { return 1 / (1 + 1 / (2 * Math.PI * fc * dt)); }
  filter(x, t, customFcMin = null) {
    if (this.xPrev === null) { this.xPrev = x; this.tPrev = t; return x; }
    const dt = Math.max(t - this.tPrev, 1e-5);
    const dx = (x - this.xPrev) / dt;
    const aD = OneEuroFilter.alpha(this.dCutoff, dt);
    const edx = aD * dx + (1 - aD) * this.dxPrev;
    const fc = (customFcMin === null ? this.fcMin : customFcMin) + this.beta * Math.abs(edx);
    const a = OneEuroFilter.alpha(fc, dt);
    const out = a * x + (1 - a) * this.xPrev;
    this.xPrev = out; this.dxPrev = edx; this.tPrev = t;
    return out;
  }
}

/** Raw pointer -> the pointer the cursor follows. Smooths harder while the fingers close for a pinch. */
export class Pointer {
  constructor() { this.fx = new OneEuroFilter(); this.fy = new OneEuroFilter(); }
  update(raw, t, preclick = false) {
    const fc = preclick ? Math.min(FC_MIN_PRECLICK, FC_MIN) : FC_MIN;
    return [this.fx.filter(raw[0], t, fc), this.fy.filter(raw[1], t, fc)];
  }
}

/** A point of the view (0..1) inside the comfort box -> a point of a w x h screen. */
export function toScreen(nx, ny, w, h, region = ACTIVE_REGION) {
  const [x0, y0, x1, y1] = region;
  const clamp = v => Math.min(Math.max(v, 0), 1);
  return [clamp((nx - x0) / (x1 - x0)) * (w - 1), clamp((ny - y0) / (y1 - y0)) * (h - 1)];
}

// ---- the state machine (fsm.py)

const ev = (kind, arg = null) => ({ kind, arg });

/** update(pose, t) -> [state, events]. Palm moves; grip clicks or drags; thumbs scroll; a held V right-clicks. */
export class AirPointFSM {
  constructor() {
    this.state = "RESTING"; this.tEnter = 0;
    this.holdPose = null; this.holdFired = false; this.holdArmed = true;
    this.scrollDir = 0; this.nextScrollT = 0;
  }
  neutral(pose) { return !GRIP.has(pose) && !(pose in SCROLL) && !(pose in HOLD_EVENTS); }
  get cursorFrozen() { return this.state === "RESTING" || this.state === "SCROLLING"; }
  to(state, t) { this.state = state; this.tEnter = t; }

  update(pose, t) {
    const events = [];
    if (pose === UNKNOWN) return [this.state, events];
    if (pose === NONE) {
      if (this.state === "DRAGGING") events.push(ev("drag_end"));
      this.to("RESTING", t);
      return [this.state, events];
    }
    const s = this.state;
    if (s === "RESTING") {
      this.to("MOVING", t);
    } else if (s === "MOVING") {
      if (this.neutral(pose)) this.holdArmed = true;
      else if (GRIP.has(pose)) this.to("PRESSING", t);
      else if (pose in SCROLL) { this.scrollDir = SCROLL[pose]; this.nextScrollT = t; this.to("SCROLLING", t); }
      else if (pose in HOLD_EVENTS && this.holdArmed) { this.holdPose = pose; this.holdFired = false; this.to("HOLD", t); }
    } else if (s === "PRESSING") {
      const held = t - this.tEnter;
      if (!GRIP.has(pose)) {
        if (held < PRESS_HOLD_S) events.push(ev("click"));
        this.to("MOVING", t);
      } else if (held >= PRESS_HOLD_S) {
        events.push(ev("drag_start"));
        this.to("DRAGGING", t);
      }
    } else if (s === "DRAGGING") {
      if (!GRIP.has(pose)) { events.push(ev("drag_end")); this.to("MOVING", t); }
    } else if (s === "HOLD") {
      if (pose !== this.holdPose) {
        if (this.neutral(pose)) this.holdArmed = true;
        this.to("MOVING", t);
      } else if (!this.holdFired && t - this.tEnter >= HOLD_FIRE_S) {
        events.push(ev(HOLD_EVENTS[pose]));
        this.holdFired = true; this.holdArmed = false;
      }
    } else if (s === "SCROLLING") {
      if (!(pose in SCROLL)) {
        this.to("MOVING", t);
      } else {
        this.scrollDir = SCROLL[pose];
        if (t >= this.nextScrollT) {
          events.push(ev("scroll", this.scrollDir * SCROLL_TICKS));
          this.nextScrollT = t + SCROLL_PERIOD_S;
        }
      }
    }
    return [this.state, events];
  }
}

/** update(xy, t, active) -> [event | null, progress 0..1]: a click when the cursor stays put for a second. */
export class DwellDetector {
  constructor(dwellS = DWELL_S, stillPx = DWELL_STILL_PX, rearmPx = DWELL_REARM_PX) {
    this.dwellS = dwellS; this.stillPx = stillPx; this.rearmPx = rearmPx;
    this.anchor = null; this.tAnchor = 0; this.clickSpot = null;
  }
  reset() { this.anchor = null; this.clickSpot = null; }
  update(xy, t, active) {
    if (!active || xy === null) { this.anchor = null; return [null, 0]; }
    if (this.clickSpot !== null) {
      if (d2(xy, this.clickSpot) < this.rearmPx) { this.anchor = null; return [null, 0]; }
      this.clickSpot = null;
    }
    if (this.anchor === null || d2(xy, this.anchor) > this.stillPx) {
      this.anchor = xy; this.tAnchor = t;
      return [null, 0];
    }
    const progress = (t - this.tAnchor) / this.dwellS;
    if (progress >= 1) { this.clickSpot = xy; this.anchor = null; return [ev("click"), 1]; }
    return [null, progress];
  }
}

// ---- the live loop (modes/live.py, without tremor filter, wheel, pause, talk or helper)

/** One frame in, what to do out: classify -> vote -> state machine -> pointer -> dwell. */
export class Pipeline {
  constructor(model, { dwell = false } = {}) {
    this.mlp = new Mlp(model);
    this.vote = new VoteSmoother(); this.fsm = new AirPointFSM();
    this.pointer = new Pointer(); this.dwell = new DwellDetector();
    this.dwellOn = dwell;
  }

  /**
   * pts: 21 [x, y, z] in the mirrored picture (0..1), or null when no hand is seen.
   * view: [w, h] of the canonical view; screen: [w, h] of what the cursor moves on.
   * -> { label, conf, state, events, cursor: [x, y] on the screen or null when it did not move, progress, ratio }
   */
  step(pts, t, view, screen) {
    let seen = null, conf = 0, preclick = false, ratio = 1;
    if (pts) {
      [seen, conf] = this.mlp.predict(pts);
      ratio = pinchRatio(pts);
      preclick = ratio < PRECLICK_RATIO;
    }
    const label = this.vote.update(seen, conf);
    let [state, events] = this.fsm.update(label, t);
    let smooth = null, cursor = null;
    if (pts) {
      const raw = [pts[MIDDLE_MCP][0] * view[0], pts[MIDDLE_MCP][1] * view[1]];
      smooth = this.pointer.update(raw, t, preclick);
      if (!this.fsm.cursorFrozen) cursor = toScreen(smooth[0] / view[0], smooth[1] / view[1], screen[0], screen[1]);
    }
    const [dwellEvent, progress] = this.dwell.update(smooth, t, this.dwellOn && state === "MOVING");
    if (dwellEvent) events = events.concat([dwellEvent]);
    return { label, conf, state, events, cursor, progress, ratio, smooth };
  }
}

// ---- pointing-test statistics (lab/stats.py): ISO 9241-411 with effective width

const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];

function trialPath(tr, w) {
  const pts = [tr.start, ...(tr.path || []).map(p => [p[1], p[2]]), tr.click];
  const straight = d2(tr.click, tr.start);
  let length = 0;
  for (let i = 1; i < pts.length; i++) length += d2(pts[i], pts[i - 1]);
  const inside = pts.map(p => d2(p, tr.dst) <= w / 2);
  let entries = inside[0] ? 1 : 0;
  for (let i = 1; i < inside.length; i++) if (!inside[i - 1] && inside[i]) entries += 1;
  return [straight > 0 ? length / straight : 1, Math.max(entries - 1, 0)];
}

/** One difficulty level -> its statistics. We = 4.133 x SD of the clicks along the task axis. */
export function sequenceStats(seq) {
  const dx = [], ae = [], mt = [], trials = [];
  for (const tr of seq.trials) {
    const axis = sub(tr.dst, tr.src), len = Math.hypot(axis[0], axis[1]);
    const u = [axis[0] / len, axis[1] / len];
    dx.push(dot(sub(tr.click, tr.dst), u));
    ae.push(dot(sub(tr.click, tr.start), u));
    mt.push((tr.t1 - tr.t0) / 1000);
    const [ratio, reentries] = trialPath(tr, seq.W);
    trials.push({ mt: mt[mt.length - 1], dx: dx[dx.length - 1], hit: Boolean(tr.hit), path_ratio: ratio, reentries });
  }
  const n = dx.length;
  let sdx = 0;
  if (n >= 2) { const m = mean(dx); sdx = Math.sqrt(dx.reduce((a, v) => a + (v - m) * (v - m), 0) / (n - 1)); }
  const flat = sdx < 1e-6;
  const we = flat ? seq.W : 4.133 * sdx;
  const aeMean = mean(ae), mtMean = mean(mt);
  const ide = aeMean > 0 ? Math.log2(aeMean / we + 1) : 0;
  return { D: seq.D, W: seq.W, n, errors: trials.filter(t => !t.hit).length, sdx, we, ae: aeMean, ide, mt: mtMean,
           tp: mtMean > 0 ? ide / mtMean : 0, flat,
           path_ratio: mean(trials.map(t => t.path_ratio)), reentries: mean(trials.map(t => t.reentries)), trials };
}

/** One block -> what the result card shows. */
export function blockSummary(block) {
  if (block.kind === "tutorial") {
    const steps = block.steps || [];
    return { kind: "tutorial", seconds: steps.reduce((a, s) => a + s.seconds, 0), steps: steps.length };
  }
  const seqs = block.sequences.filter(s => s.trials.length).map(sequenceStats);
  const trials = seqs.reduce((a, s) => a + s.n, 0);
  return { kind: "test", tp: mean(seqs.map(s => s.tp)), trials,
           error_rate: seqs.reduce((a, s) => a + s.errors, 0) / trials,
           mt: seqs.reduce((a, s) => a + s.mt * s.n, 0) / trials,
           path_ratio: mean(seqs.map(s => s.path_ratio)), reentries: mean(seqs.map(s => s.reentries)),
           flat: seqs.some(s => s.flat),
           sequences: seqs.map(({ trials: _, ...rest }) => rest) };
}

// ---- parity: the same inputs Python was given must give the same answers here

/** Run every check in a fixture written by `python -m airpoint.export_web`. -> { checks, failures: [text] } */
export function checkParity(model, fx, tol = 1e-9) {
  const failures = [];
  let checks = 0;
  const near = (a, b, what, eps = tol) => {
    checks += 1;
    if (!(Math.abs(a - b) <= eps * Math.max(1, Math.abs(b)))) failures.push(`${what}: ${a} != ${b}`);
  };
  const same = (a, b, what) => { checks += 1; if (JSON.stringify(a) !== JSON.stringify(b)) failures.push(`${what}: ${JSON.stringify(a)} != ${JSON.stringify(b)}`); };

  const mlp = new Mlp(model);
  fx.hands.forEach((h, i) => {
    const f = normalize63(h.pts);
    h.features.forEach((v, k) => near(f[k], v, `hand ${i} feature ${k}`));
    near(pinchRatio(h.pts), h.pinch, `hand ${i} pinch ratio`);
    near(palmPx(h.pts, 640, 360), h.palm_px, `hand ${i} palm size`);
    const p = mlp.proba(h.pts);
    h.proba.forEach((v, k) => near(p[k], v, `hand ${i} probability ${k}`));
    same(mlp.predict(h.pts)[0], h.label, `hand ${i} label`);
  });

  const vote = new VoteSmoother();
  fx.vote.steps.forEach(([label, conf], i) => same(vote.update(label, conf), fx.vote.out[i], `vote step ${i}`));

  const f1 = new OneEuroFilter(fx.filter.fc_min, fx.filter.beta);
  fx.filter.steps.forEach(([x, t, fc], i) => near(f1.filter(x, t, fc), fx.filter.out[i], `filter step ${i}`));

  const pointer = new Pointer();
  fx.pointer.steps.forEach(([raw, t, pre], i) => {
    const out = pointer.update(raw, t, pre);
    near(out[0], fx.pointer.out[i][0], `pointer x ${i}`); near(out[1], fx.pointer.out[i][1], `pointer y ${i}`);
  });

  const fsm = new AirPointFSM();
  fx.fsm.steps.forEach(([pose, t], i) => {
    const [state, events] = fsm.update(pose, t);
    same([state, events.map(e => [e.kind, e.arg]), fsm.cursorFrozen], fx.fsm.out[i], `state machine step ${i}`);
  });

  const dwell = new DwellDetector();
  fx.dwell.steps.forEach(([xy, t, active], i) => {
    const [event, progress] = dwell.update(xy, t, active);
    same(event ? event.kind : null, fx.dwell.out[i][0], `dwell event ${i}`);
    near(progress, fx.dwell.out[i][1], `dwell progress ${i}`);
  });

  fx.to_screen.forEach((p, i) => {
    const out = toScreen(p.n[0], p.n[1], 1000, 500);
    near(out[0], p.out[0], `to screen x ${i}`); near(out[1], p.out[1], `to screen y ${i}`);
  });

  const deep = (a, b, what) => {
    if (typeof b === "number") return near(a, b, what);
    if (b === null || typeof b !== "object") return same(a, b, what);
    if (Array.isArray(b)) { same(Array.isArray(a) && a.length, b.length, `${what} length`); return b.forEach((v, i) => deep(a && a[i], v, `${what}[${i}]`)); }
    Object.keys(b).forEach(k => deep(a && a[k], b[k], `${what}.${k}`));
  };
  deep(blockSummary(fx.stats.block), fx.stats.summary, "block summary");
  fx.stats.sequences.forEach((s, i) => deep(sequenceStats(fx.stats.block.sequences[i]), s, `sequence ${i}`));
  deep(sequenceStats(fx.stats.flat.seq), fx.stats.flat.stats, "one-click sequence");

  // one whole session: the same frames the real Python live loop was given
  const pipe = new Pipeline(model);
  let down = false;
  fx.session.frames.forEach((frame, i) => {
    const out = pipe.step(frame.pts, frame.t, fx.session.view, fx.session.screen);
    const calls = [];
    if (out.cursor) calls.push(["move", Math.round(out.cursor[0]), Math.round(out.cursor[1])]);
    for (const e of out.events) {
      if (e.kind === "click" || e.kind === "right_click") calls.push([e.kind]);
      else if (e.kind === "drag_start") { calls.push(["down"]); down = true; }
      else if (e.kind === "drag_end") { if (down) calls.push(["up"]); down = false; }
      else if (e.kind === "scroll") calls.push(["scroll", e.arg]);
    }
    const want = fx.session.out[i];
    same(out.label, want.label, `session frame ${i} pose`);
    same(out.state, want.state, `session frame ${i} state`);
    same(calls.map(c => c[0]), want.calls.map(c => c[0]), `session frame ${i} actions`);
    calls.forEach((c, k) => {
      const w = want.calls[k];
      if (w && c[0] === "move" && w[0] === "move") { near(c[1], w[1], `session frame ${i} cursor x`, 1.0001 / Math.max(1, Math.abs(w[1]))); near(c[2], w[2], `session frame ${i} cursor y`, 1.0001 / Math.max(1, Math.abs(w[2]))); }
      else if (w) same(c, w, `session frame ${i} action ${k}`);
    });
  });
  return { checks, failures };
}
