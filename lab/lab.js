// Shared by the three lab pages. No framework, no build step.
const Lab = {
  KEY: "airpoint.lab.who",

  async api(path, body) {
    const r = await fetch(path, body === undefined ? {} : {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
    return data;
  },

  who() {
    try { return JSON.parse(localStorage.getItem(this.KEY)) || null; } catch (e) { return null; }
  },
  saveWho(who) { localStorage.setItem(this.KEY, JSON.stringify(who)); },

  // AirPoint's live settings, or {running: false} when it is not running or the server is away.
  async state() {
    try { return await this.api("/api/state"); } catch (e) { return { running: false, state: null }; }
  },

  // One line a person can check against what they see in the AirPoint window.
  describe(s) {
    if (!s) return "AirPoint is not running";
    const bits = [s.mouse ? "mouse ON" : "mouse OFF", s.dwell ? "dwell click" : "sign click", `filter ${s.filter}`];
    if (s.sim && s.sim.on) bits.push(`simulated tremor ${s.sim.amp.toFixed(2)} palm @ ${s.sim.hz.toFixed(1)} Hz`);
    if (s.locked) bits.push(`tremor locked ${s.lock_hz.toFixed(1)} Hz`);
    return bits.join(" · ");
  },

  // Text typed by a person goes into the page through this.
  esc(s) {
    return String(s === null || s === undefined ? "" : s).replace(/[&<>"']/g,
      c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  },

  fmt(x, digits = 2) { return (x === null || x === undefined || Number.isNaN(x)) ? "–" : Number(x).toFixed(digits); },
  pct(x) { return (x === null || x === undefined) ? "–" : (100 * x).toFixed(0) + "%"; },
  el(id) { return document.getElementById(id); },

  // Pages other than the home need to know who is testing; send them back if not.
  requireWho() {
    const who = this.who();
    if (!who || !who.participant) { location.href = "index.html"; return null; }
    return who;
  },
};
