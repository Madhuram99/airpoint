// A pointer that is not the mouse: it turns AirPoint's cursor position and events into ordinary
// mouse events inside a same-origin frame, so pages written for a mouse work unchanged.

const SCROLL_PX = 40, DOUBLE_MS = 500, DOUBLE_PX = 14;

export class VirtualPointer {
  /** frame: the <iframe>; cursor: the element drawn as the cursor (positioned over the frame). */
  constructor(frame, cursor) {
    this.frame = frame; this.cursor = cursor;
    this.x = 0; this.y = 0; this.over = null; this.pressed = null; this.lastClick = null;
    frame.addEventListener("load", () => { this.over = null; this.pressed = null; });
  }

  get win() { return this.frame.contentWindow; }
  get doc() { return this.frame.contentDocument; }

  target() {
    const d = this.doc;
    return d ? (d.elementFromPoint(this.x, this.y) || d.body) : null;
  }

  fire(el, type, extra = {}) {
    if (!el || !this.win) return;
    const bubbles = type !== "mouseenter" && type !== "mouseleave";
    el.dispatchEvent(new this.win.MouseEvent(type, {
      bubbles, cancelable: true, view: this.win, clientX: this.x, clientY: this.y, screenX: this.x, screenY: this.y,
      button: 0, buttons: this.pressed ? 1 : 0, ...extra }));
  }

  /** Move to (x, y) in the frame's own coordinates. */
  move(x, y) {
    this.x = x; this.y = y;
    this.cursor.style.transform = `translate(${x}px, ${y}px)`;
    this.cursor.classList.remove("away");
    const el = this.target();
    if (el !== this.over) {
      const chain = e => { const out = []; for (; e; e = e.parentElement) out.push(e); return out; };
      const was = chain(this.over && this.over.isConnected ? this.over : null), now = chain(el);
      if (was.length) this.fire(was[0], "mouseout");
      was.filter(e => !now.includes(e)).forEach(e => this.fire(e, "mouseleave"));
      if (el) this.fire(el, "mouseover");
      now.filter(e => !was.includes(e)).reverse().forEach(e => this.fire(e, "mouseenter"));
      this.over = el;
    }
    this.fire(el, "mousemove");
  }

  away() { this.cursor.classList.add("away"); }

  down() {
    const el = this.target();
    this.pressed = el;
    this.fire(el, "mousedown", { buttons: 1 });
    this.cursor.classList.add("down");
  }

  up() {
    const el = this.target(), pressed = this.pressed;
    this.pressed = null;
    this.fire(el, "mouseup");
    this.cursor.classList.remove("down");
    if (pressed && pressed === el) this.fire(el, "click", { detail: 1 });
  }

  click() {
    const el = this.target(), now = performance.now();
    const again = this.lastClick && now - this.lastClick.t < DOUBLE_MS &&
                  Math.hypot(this.x - this.lastClick.x, this.y - this.lastClick.y) < DOUBLE_PX;
    this.pressed = el;
    this.fire(el, "mousedown", { buttons: 1, detail: again ? 2 : 1 });
    this.pressed = null;
    if (el && el.focus && el.tabIndex >= 0) el.focus({ preventScroll: true });
    this.fire(el, "mouseup", { detail: again ? 2 : 1 });
    this.fire(el, "click", { detail: again ? 2 : 1 });
    if (again) this.fire(el, "dblclick", { detail: 2 });
    this.lastClick = again ? null : { t: now, x: this.x, y: this.y };
    this.pulse();
  }

  rightClick() {
    const el = this.target();
    this.fire(el, "mousedown", { button: 2, buttons: 2 });
    this.fire(el, "mouseup", { button: 2 });
    this.fire(el, "contextmenu", { button: 2 });
    this.pulse("right");
  }

  /** ticks > 0 scrolls up, as a mouse wheel does. */
  scroll(ticks) {
    let el = this.target();
    for (; el && el !== this.doc.body; el = el.parentElement) {
      const flow = this.win.getComputedStyle(el).overflowY;
      if ((flow === "auto" || flow === "scroll") && el.scrollHeight > el.clientHeight + 1) { el.scrollTop -= ticks * SCROLL_PX; return; }
    }
    this.win.scrollBy(0, -ticks * SCROLL_PX);
  }

  pulse(kind = "") {
    this.cursor.classList.remove("pulse", "right");
    void this.cursor.offsetWidth;                      // restart the animation
    this.cursor.classList.add("pulse");
    if (kind) this.cursor.classList.add(kind);
  }

  /** Carry out the events of one frame of the pipeline. */
  apply(events) {
    for (const e of events) {
      if (e.kind === "click") this.click();
      else if (e.kind === "right_click") this.rightClick();
      else if (e.kind === "drag_start") this.down();
      else if (e.kind === "drag_end") { if (this.pressed) this.up(); }
      else if (e.kind === "scroll") this.scroll(e.arg);
    }
  }
}
