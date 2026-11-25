// A robot hand for checking the page without a camera: it feeds recorded hands into the same
// pipeline the camera feeds, and steers them by looking at the practice page as a person would.
//   try/?replay&robot=tutorial     try/?replay&robot=test
// What it proves: recorded hands -> signs -> virtual pointer -> the practice pages -> the score.
// What it cannot prove: that the camera and the hand tracker see a real hand.

const sleep = ms => new Promise(r => setTimeout(r, ms));
const FRAME_MS = 33;

export async function run(debug, what, log = () => {}) {
  const { AP, onHand, frame } = debug;
  const fixture = await (await fetch("parity.json")).json();
  const hands = {};
  for (const h of fixture.hands) if (!hands[h.label]) hands[h.label] = h.pts;
  const [x0, y0, x1, y1] = AP.ACTIVE_REGION;
  let at = [frame.clientWidth / 2, frame.clientHeight / 2];
  const state = { what, done: false, steps: [], error: null };
  window.AirPointRobot = state;

  /** The recorded hand `pose`, moved so that the cursor it produces sits at (cx, cy) in the frame. */
  const place = (pose, cx, cy) => {
    const pts = hands[pose], m = pts[AP.MIDDLE_MCP];
    const nx = x0 + cx / (frame.clientWidth - 1) * (x1 - x0), ny = y0 + cy / (frame.clientHeight - 1) * (y1 - y0);
    return pts.map(p => [p[0] - m[0] + nx, p[1] - m[1] + ny, p[2]]);
  };
  const hold = async (pose, seconds) => {
    for (const end = performance.now() + seconds * 1000; performance.now() < end;) {
      onHand(pose ? place(pose, at[0], at[1]) : null, performance.now() / 1000);
      await sleep(FRAME_MS);
    }
  };
  const glide = async (pose, to, seconds = 0.6) => {
    const from = at, start = performance.now();
    for (;;) {
      const k = Math.min((performance.now() - start) / (seconds * 1000), 1), e = k * k * (3 - 2 * k);
      at = [from[0] + (to[0] - from[0]) * e, from[1] + (to[1] - from[1]) * e];
      onHand(place(pose, at[0], at[1]), performance.now() / 1000);
      if (k >= 1) break;
      await sleep(FRAME_MS);
    }
    await hold(pose, 0.35);                      // let the smoothing settle on the spot
  };
  const centre = el => { const r = el.getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; };
  const click = async to => { await glide("palm", to); await hold("pinch", 0.2); await hold("palm", 0.3); };
  const doc = () => frame.contentDocument;
  const until = async (test, seconds, what) => {
    for (const end = performance.now() + seconds * 1000; performance.now() < end;) { if (test()) return; await hold("palm", 0.1); }
    throw new Error(`timed out waiting for ${what}`);
  };

  async function tutorial() {
    await until(() => doc().querySelector(".dot"), 5, "the tutorial");
    const title = () => doc().getElementById("title").textContent;
    for (const dot of [...doc().querySelectorAll(".dot")]) { await glide("palm", centre(dot)); await hold("palm", 0.6); }
    await until(() => title().startsWith("2."), 3, "step 2"); state.steps.push("move");
    for (const tile of [...doc().querySelectorAll(".tile")]) await click(centre(tile));
    await until(() => title().startsWith("3."), 3, "step 3"); state.steps.push("click");
    await glide("palm", centre(doc().querySelector(".tile")));
    for (let i = 0; i < 2; i++) { await hold("pinch", 0.14); await hold("palm", 0.14); }
    await until(() => title().startsWith("4."), 3, "step 4"); state.steps.push("double");
    await glide("palm", centre(doc().querySelector(".block")));
    await hold("pinch", 0.8);                                    // held: the block is picked up
    await glide("pinch", centre(doc().querySelector(".zone")), 1.0);
    await hold("palm", 0.4);                                     // open hand: dropped
    await until(() => title().startsWith("5."), 3, "step 5"); state.steps.push("drag");
    const list = doc().querySelector(".scroller"), goal = doc().querySelector(".goal");
    await glide("palm", centre(list));
    const inView = () => { const g = goal.getBoundingClientRect(), l = list.getBoundingClientRect(); return g.top > l.top + 10 && g.bottom < l.bottom - 10; };
    for (let i = 0; i < 80 && !inView(); i++) await hold("thumb_down", 0.1);
    await hold("palm", 0.3);
    await click(centre(goal));
    await until(() => title().startsWith("6."), 3, "step 6"); state.steps.push("scroll");
    await glide("palm", centre(doc().querySelector(".tile")));
    await hold("v_sign", 0.6);
    await hold("palm", 0.3);
    await until(() => !doc().getElementById("finished").classList.contains("hidden"), 3, "the end of the tutorial");
    state.steps.push("right");
    state.result = { total: doc().getElementById("total").textContent, done: doc().getElementById("done-count").textContent,
                     note: doc().getElementById("save-note").textContent };
  }

  async function test() {
    await until(() => doc().getElementById("start"), 5, "the pointing test");
    await click(centre(doc().getElementById("start")));
    const win = frame.contentWindow, canvas = doc().getElementById("stage"), ctx = canvas.getContext("2d");
    const dpr = win.devicePixelRatio || 1, vw = win.innerWidth, vh = win.innerHeight, BAR = 56;
    const scale = Math.min(1, (Math.min(vw, vh - BAR) - 24) / (640 + 48)), cx = vw / 2, cy = BAR + (vh - BAR) / 2;
    const spots = [];
    for (const d of [320, 480, 640]) for (let i = 0; i < 9; i++) {
      const a = -Math.PI / 2 + 2 * Math.PI * i / 9;
      spots.push([cx + d * scale / 2 * Math.cos(a), cy + d * scale / 2 * Math.sin(a)]);
    }
    const lit = () => spots.find(([x, y]) => {                   // the lit target is the filled cyan one
      const p = ctx.getImageData(Math.round(x * dpr), Math.round(y * dpr), 1, 1).data;
      return p[3] > 120 && p[2] > 180 && p[1] > 160 && p[0] < 110;
    });
    const result = () => !doc().getElementById("result").classList.contains("hidden");
    let clicks = 0;
    while (!result() && clicks < 60) {
      const spot = lit();
      if (!spot) { await hold("palm", 0.1); continue; }
      await click(spot);
      clicks += 1;
    }
    await until(result, 5, "the result");
    state.result = { clicks, tp: doc().getElementById("r-tp").textContent, missed: doc().getElementById("r-err").textContent,
                     seconds: doc().getElementById("r-mt").textContent, who: doc().getElementById("result-condition").textContent,
                     levels: doc().getElementById("r-levels").rows.length };
  }

  try { await (what === "test" ? test() : tutorial()); }
  catch (e) { state.error = String(e.message || e); }
  state.done = true;
  log(state);
  return state;
}
