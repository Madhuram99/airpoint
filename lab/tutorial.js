// Six short steps, one per mouse action. Each step mounts itself into the arena and calls done().
(function () {
  const who = Lab.requireWho();
  if (!who) return;

  const HOVER_MS = 350;
  const arena = Lab.el("arena");
  const make = (cls, text, style) => {
    const e = document.createElement(cls === "tile" ? "button" : "div");
    e.className = cls; e.textContent = text; Object.assign(e.style, style || {});
    arena.appendChild(e);
    return e;
  };
  const allDone = (items, done) => { if (items.every(i => i.classList.contains("done"))) done(); };

  const STEPS = [
    { name: "move", title: "1. Move", text: "Rest the cursor on each circle until it turns green.",
      hint: "AirPoint: an open palm moves the cursor.",
      mount(done) {
        const dots = [["8%", "30%"], ["44%", "55%"], ["80%", "20%"]].map(([l, t], i) => make("dot", String(i + 1), { left: l, top: t }));
        dots.forEach(d => {
          let timer = null;
          d.addEventListener("mouseenter", () => { timer = setTimeout(() => { d.classList.add("done"); allDone(dots, done); }, HOVER_MS); });
          d.addEventListener("mouseleave", () => clearTimeout(timer));
        });
      } },
    { name: "click", title: "2. Click", text: "Click all three buttons.",
      hint: "AirPoint: pinch or make a fist, then open the hand. With dwell on, just hold still.",
      mount(done) {
        const tiles = [["8%", "20%"], ["40%", "58%"], ["72%", "25%"]].map(([l, t], i) => make("tile", "click " + (i + 1), { left: l, top: t }));
        tiles.forEach(b => b.addEventListener("click", () => { b.classList.add("done"); allDone(tiles, done); }));
      } },
    { name: "double", title: "3. Double-click", text: "Double-click the box.",
      hint: "AirPoint: two quick pinches.",
      mount(done) {
        const b = make("tile", "double-click", { left: "38%", top: "38%", width: "200px", height: "120px" });
        b.addEventListener("dblclick", () => { b.classList.add("done"); done(); });
      } },
    { name: "drag", title: "4. Drag", text: "Drag the purple block into the dashed zone.",
      hint: "AirPoint: pinch or fist and keep it held, move, then open the hand to drop.",
      mount(done) {
        const zone = make("zone", "drop here"), block = make("block", "drag");
        const home = { left: block.offsetLeft, top: block.offsetTop };
        let grab = null;
        block.addEventListener("mousedown", e => {
          if (e.button !== 0) return;
          grab = { dx: e.clientX - block.offsetLeft, dy: e.clientY - block.offsetTop };
          e.preventDefault();
        });
        const move = e => {
          if (!grab) return;
          block.style.left = Math.max(0, Math.min(arena.clientWidth - block.offsetWidth, e.clientX - grab.dx)) + "px";
          block.style.top = Math.max(0, Math.min(arena.clientHeight - block.offsetHeight, e.clientY - grab.dy)) + "px";
        };
        const up = () => {
          if (!grab) return;
          grab = null;
          const bx = block.offsetLeft + block.offsetWidth / 2, by = block.offsetTop + block.offsetHeight / 2;
          const inside = bx > zone.offsetLeft && bx < zone.offsetLeft + zone.offsetWidth &&
                         by > zone.offsetTop && by < zone.offsetTop + zone.offsetHeight;
          if (inside) { zone.classList.add("done"); cleanup(); done(); }
          else { block.style.left = home.left + "px"; block.style.top = home.top + "px"; }
        };
        const cleanup = () => { document.removeEventListener("mousemove", move); document.removeEventListener("mouseup", up); };
        document.addEventListener("mousemove", move);
        document.addEventListener("mouseup", up);
        return cleanup;
      } },
    { name: "scroll", title: "5. Scroll", text: "Scroll down to the bright line and click it.",
      hint: "AirPoint: thumb down scrolls down, thumb up scrolls up. Keep the cursor over the list.",
      mount(done) {
        const list = make("scroller", "");
        for (let i = 1; i <= 60; i++) {
          const line = document.createElement("div");
          line.textContent = i === 44 ? "line 44: click me" : "line " + i;
          if (i === 44) { line.className = "goal"; line.addEventListener("click", done); }
          list.appendChild(line);
        }
      } },
    { name: "right", title: "6. Right-click", text: "Right-click the box.",
      hint: "AirPoint: hold a V-sign for a moment.",
      mount(done) {
        const b = make("tile", "right-click", { left: "38%", top: "38%", width: "200px", height: "120px" });
        b.addEventListener("contextmenu", e => { e.preventDefault(); b.classList.add("done"); done(); });
      } },
  ];

  const results = [];
  let i = -1, startedAt = 0, cleanup = null, finishedStep = false;
  arena.addEventListener("contextmenu", e => e.preventDefault());   // a missed right-click must not open the browser menu

  function show() {
    Lab.el("steps").innerHTML = STEPS.map((_, n) => `<span class="${n < i ? "done" : n === i ? "now" : ""}"></span>`).join("");
    const s = STEPS[i];
    Lab.el("title").textContent = s.title; Lab.el("text").textContent = s.text; Lab.el("hint").textContent = s.hint;
    arena.innerHTML = "";
    finishedStep = false;
    startedAt = performance.now();
    cleanup = s.mount(() => next(false)) || null;
  }

  function next(skipped) {
    if (finishedStep) return;                 // a step reports once, however many events fire
    finishedStep = true;
    if (cleanup) cleanup();
    results.push({ name: STEPS[i].name, seconds: (performance.now() - startedAt) / 1000, skipped: skipped });
    i += 1;
    if (i < STEPS.length) setTimeout(show, skipped ? 0 : 450); else finish();
  }

  async function finish() {
    Lab.el("lesson").classList.add("hidden");
    Lab.el("finished").classList.remove("hidden");
    const total = results.reduce((a, r) => a + r.seconds, 0);
    Lab.el("total").textContent = Lab.fmt(total, 0);
    Lab.el("done-count").textContent = `${results.filter(r => !r.skipped).length} / ${STEPS.length}`;
    Lab.el("rows").innerHTML = results.map(r =>
      `<tr><td>${r.name}</td><td class="num">${Lab.fmt(r.seconds, 1)}</td><td>${r.skipped ? "skipped" : "done"}</td></tr>`).join("");
    const s = await Lab.state();
    try {
      const out = await Lab.api("/api/blocks", { kind: "tutorial", participant: who.participant, device: who.device, hand: who.hand,
        started_at: new Date().toISOString(), airpoint_start: s.running ? s.state : null, steps: results });
      Lab.el("save-note").textContent = out.note || `Saved for ${who.participant} (${out.condition}).`;
    } catch (e) {
      Lab.el("save-note").textContent = "Could not save: " + e.message;
    }
  }

  setInterval(() => {
    if (i >= 0 && i < STEPS.length) Lab.el("clock").textContent = `${((performance.now() - startedAt) / 1000).toFixed(0)} s on this step`;
  }, 500);
  Lab.el("skip").addEventListener("click", () => next(true));
  i = 0;
  show();
})();
