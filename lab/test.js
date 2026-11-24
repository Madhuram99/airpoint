// ISO 9241-411 multi-directional tapping. The page records raw events only;
// the server (Python) does every calculation.
(function () {
  const who = Lab.requireWho();
  if (!who) return;

  const N = 9;                                        // targets on the ring
  const LEVELS = [[320, 96], [480, 64], [640, 48]];    // [distance, width] in px before scaling
  const BAR = 56, REFRACTORY_MS = 120, PATH_EVERY_MS = 8, PATH_MAX = 1500;

  const canvas = Lab.el("stage"), ctx = canvas.getContext("2d");
  let vw = 0, vh = 0, scale = 1, cx = 0, cy = 0;
  let phase = "intro";                                 // intro | ready | running | saving | done
  let level = 0, k = 0, D = 0, W = 0, targets = [], seq = null;
  let prevClick = null, t0 = 0, path = [], lastSel = 0, lastSample = 0, flash = null;

  const order = [0, 1, 2].sort(() => Math.random() - 0.5);
  const block = {
    kind: "test", participant: who.participant, device: who.device, hand: who.hand,
    started_at: new Date().toISOString(), level_order: order, sequences: [],
    airpoint_start: null, airpoint_end: null,
  };

  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const idx = step => (step * ((N + 1) / 2)) % N;     // step 0 is the start target, then across the ring

  function layout() {
    vw = window.innerWidth; vh = window.innerHeight;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(vw * dpr); canvas.height = Math.round(vh * dpr);
    canvas.style.width = vw + "px"; canvas.style.height = vh + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    scale = Math.min(1, (Math.min(vw, vh - BAR) - 24) / (LEVELS[2][0] + LEVELS[2][1]));
    cx = vw / 2; cy = BAR + (vh - BAR) / 2;
    block.viewport = { w: vw, h: vh, dpr: dpr };
    block.scale = scale;
  }

  function startLevel() {
    const [d0, w0] = LEVELS[order[level]];
    D = d0 * scale; W = w0 * scale;
    targets = [];
    for (let i = 0; i < N; i++) {
      const a = -Math.PI / 2 + 2 * Math.PI * i / N;
      targets.push([cx + D / 2 * Math.cos(a), cy + D / 2 * Math.sin(a)]);
    }
    seq = { D: D, W: W, n: N, nominal: { D: d0, W: w0 }, trials: [] };
    k = 0; phase = "ready";
  }

  async function finish() {
    phase = "saving";
    const s = await Lab.state();
    block.airpoint_end = s.running ? s.state : null;
    let out;
    try { out = await Lab.api("/api/blocks", block); } catch (e) {
      Lab.el("r-note").textContent = "Could not save: " + e.message;
      Lab.el("result").classList.remove("hidden"); phase = "done"; return;
    }
    const r = out.summary;
    Lab.el("result-condition").textContent = `${block.participant} · ${out.condition}`;
    Lab.el("r-tp").textContent = Lab.fmt(r.tp);
    Lab.el("r-err").textContent = Lab.pct(r.error_rate);
    Lab.el("r-mt").textContent = Lab.fmt(r.mt);
    Lab.el("r-levels").innerHTML = r.sequences.map((q, i) =>
      `<tr><td>${i + 1}</td><td class="num">${Lab.fmt(q.D, 0)} px</td><td class="num">${Lab.fmt(q.W, 0)} px</td>` +
      `<td class="num">${Lab.fmt(q.tp)}</td><td class="num">${q.errors} / ${q.n}</td><td class="num">${Lab.fmt(q.path_ratio)}</td></tr>`).join("");
    Lab.el("r-note").textContent = r.flat
      ? "One level had clicks with no spread along the movement, so its nominal width was used."
      : "Throughput uses the spread of your clicks, not the drawn target size (ISO 9241-411).";
    Lab.el("result").classList.remove("hidden");
    phase = "done";
  }

  canvas.addEventListener("mousedown", e => {
    if (e.button !== 0) return;
    const now = performance.now(), p = [e.clientX, e.clientY];
    if (phase === "ready") {
      if (dist(p, targets[0]) > W / 2) return;        // a level starts on its top target
      prevClick = p; t0 = now; lastSel = now; path = []; k = 1; phase = "running";
      return;
    }
    if (phase !== "running" || now - lastSel < REFRACTORY_MS) return;
    const dst = targets[idx(k)], src = targets[idx(k - 1)];
    const hit = dist(p, dst) <= W / 2;
    seq.trials.push({ src: src, dst: dst, start: prevClick, click: p, t0: t0, t1: now, hit: hit, path: path });
    flash = { at: now, hit: hit, p: p };
    prevClick = p; t0 = now; lastSel = now; path = []; k += 1;
    if (k > N) {
      block.sequences.push(seq);
      level += 1;
      if (level < LEVELS.length) startLevel(); else finish();
    }
  });

  canvas.addEventListener("mousemove", e => {
    if (phase !== "running") return;
    const now = performance.now();
    if (now - lastSample < PATH_EVERY_MS || path.length >= PATH_MAX) return;
    lastSample = now;
    path.push([now, e.clientX, e.clientY]);
  });

  function draw(now) {
    ctx.clearRect(0, 0, vw, vh);
    if (phase === "ready" || phase === "running") {
      const lit = phase === "ready" ? 0 : idx(k);
      targets.forEach((t, i) => {
        ctx.beginPath(); ctx.arc(t[0], t[1], W / 2, 0, 2 * Math.PI);
        if (i === lit) {
          const pulse = phase === "ready" ? 0.65 + 0.35 * Math.sin(now / 220) : 1;
          ctx.fillStyle = `rgba(34, 211, 238, ${pulse})`; ctx.fill();
        } else {
          ctx.strokeStyle = "rgba(148, 163, 184, 0.35)"; ctx.lineWidth = 2; ctx.stroke();
        }
      });
      ctx.fillStyle = "#94a3b8"; ctx.font = "600 16px -apple-system, sans-serif"; ctx.textAlign = "center";
      if (phase === "ready") ctx.fillText(`Level ${level + 1} of ${LEVELS.length}: click the lit circle at the top to start`, cx, cy);
      Lab.el("progress").textContent = `level ${level + 1} / ${LEVELS.length} · target ${Math.min(k, N)} / ${N}`;
    }
    if (flash && now - flash.at < 320) {
      const a = 1 - (now - flash.at) / 320;
      ctx.beginPath(); ctx.arc(flash.p[0], flash.p[1], 10 + 18 * (1 - a), 0, 2 * Math.PI);
      ctx.strokeStyle = flash.hit ? `rgba(52, 211, 153, ${a})` : `rgba(248, 113, 113, ${a})`; ctx.lineWidth = 3; ctx.stroke();
    }
    if (phase === "saving") {
      ctx.fillStyle = "#94a3b8"; ctx.font = "600 18px -apple-system, sans-serif"; ctx.textAlign = "center";
      ctx.fillText("Saving…", cx, cy);
    }
    requestAnimationFrame(draw);
  }

  Lab.el("intro-who").textContent = `${who.participant} · ${who.device} · ${who.hand} hand`;
  Lab.el("start").addEventListener("click", async () => {
    const s = await Lab.state();
    block.airpoint_start = s.running ? s.state : null;
    Lab.el("condition").textContent = who.device === "airpoint" ? Lab.describe(block.airpoint_start) : who.device;
    Lab.el("intro").classList.add("hidden");
    layout();                                          // geometry is fixed from here to the end of the block
    startLevel();
  });
  window.addEventListener("resize", () => { if (phase === "intro") layout(); });
  layout();
  requestAnimationFrame(draw);
})();
