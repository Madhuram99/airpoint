// The try-it page: camera -> MediaPipe hand landmarks -> AirPoint's pipeline -> a virtual pointer
// inside the practice frame. Nothing leaves the page.
import * as AP from "./airpoint.js";
import { VirtualPointer } from "./pointer.js";

const CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3";      // same version as the Python package
const HAND_MODEL = "../hand_landmarker.task";
const BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
               [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];
const TIPS = {
  none: "No hand in view. Hold your open hand up to the camera.",
  unknown: "Hold the sign a little steadier.",
  RESTING: "Hold your open hand up to the camera.",
  MOVING: "Open palm moves the cursor. Pinch to click.",
  PRESSING: "Open your hand to click, or keep holding to drag.",
  DRAGGING: "Dragging: open your hand to drop.",
  HOLD: "Right-click.",
  SCROLLING: "Scrolling: back to an open palm to stop.",
};

const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
const video = $("video"), overlay = $("overlay"), frame = $("practice");
const pointer = new VirtualPointer(frame, $("cursor"));
const ringLength = 2 * Math.PI * 15;
let pipeline = null, tracker = null, lastT = null, fps = 0;

// The practice pages have no server here: they ask this page (see index.html), and it scores them itself.
window.AirPointStatic.score = block => ({ summary: AP.blockSummary(block), condition: "AirPoint in your browser",
                                          note: "Not saved anywhere: this result is only on your screen." });

function status(text, kind = "") { $("status").textContent = text; $("status").className = kind; }

function drawHand(pts) {
  const w = overlay.clientWidth, h = overlay.clientHeight, dpr = window.devicePixelRatio || 1;
  if (overlay.width !== Math.round(w * dpr)) { overlay.width = Math.round(w * dpr); overlay.height = Math.round(h * dpr); }
  const ctx = overlay.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!pts) return;
  ctx.strokeStyle = "#e879f9"; ctx.lineWidth = 2;
  for (const [a, b] of BONES) { ctx.beginPath(); ctx.moveTo(pts[a][0] * w, pts[a][1] * h); ctx.lineTo(pts[b][0] * w, pts[b][1] * h); ctx.stroke(); }
  ctx.fillStyle = "#22d3ee";
  for (const p of pts) { ctx.beginPath(); ctx.arc(p[0] * w, p[1] * h, 2.5, 0, 2 * Math.PI); ctx.fill(); }
}

/** One frame: `pts` are 21 [x, y, z] in the mirrored picture, or null. `aspect` = picture height / width. */
function onHand(pts, t, aspect = 9 / 16) {
  if (!pipeline) return null;
  const view = [AP.CANON_W, AP.CANON_W * aspect], screen = [frame.clientWidth, frame.clientHeight];
  const out = pipeline.step(pts, t, view, screen);
  if (out.cursor) pointer.move(out.cursor[0], out.cursor[1]);
  else if (!pts) pointer.away();
  pointer.apply(out.events);
  $("ring").firstElementChild.style.strokeDashoffset = String(ringLength * (1 - Math.min(out.progress, 1)));
  drawHand(pts);
  if (lastT !== null && t > lastT) fps = fps ? 0.9 * fps + 0.1 / (t - lastT) : 1 / (t - lastT);
  lastT = t;
  $("sign").textContent = pts ? `${out.label} ${(out.conf * 100).toFixed(0)}%` : "no hand";
  $("doing").textContent = out.state.toLowerCase();
  $("fps").textContent = fps ? `${fps.toFixed(0)} frames a second` : "–";
  $("tip").textContent = TIPS[out.label] || TIPS[out.state] || "";
  return out;
}

async function loadEngine() {
  status("loading hand tracking (about 8 MB, once)…");
  const { FilesetResolver, HandLandmarker } = await import(`${CDN}/vision_bundle.mjs`);
  const vision = await FilesetResolver.forVisionTasks(`${CDN}/wasm`);
  const options = delegate => ({
    baseOptions: { modelAssetPath: HAND_MODEL, delegate }, runningMode: "VIDEO", numHands: 1,
    minHandDetectionConfidence: 0.7, minHandPresenceConfidence: 0.7, minTrackingConfidence: 0.7 });
  try { tracker = await HandLandmarker.createFromOptions(vision, options("GPU")); }
  catch (e) { tracker = await HandLandmarker.createFromOptions(vision, options("CPU")); }
  return tracker;
}

function detect(source, nowMs) {
  const result = tracker.detectForVideo(source, nowMs);
  const hand = result.landmarks && result.landmarks[0];
  return hand ? hand.map(p => [1 - p.x, p.y, p.z]) : null;            // mirrored, as the desktop app's camera is
}

async function startCamera() {
  $("start").disabled = true;
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error("this browser gives pages no camera here (it needs https or localhost)");
    if (!tracker) await loadEngine();
    status("asking for the camera…");
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    video.srcObject = stream;
    await video.play();
  } catch (e) {
    status(`could not start: ${e.message || e.name}`, "bad");
    $("welcome-note").textContent = e.name === "NotAllowedError"
      ? "The camera was not allowed. Allow it for this page in the address bar, then press Start again."
      : `Could not start: ${e.message || e.name}`;
    $("start").disabled = false;
    return;
  }
  $("welcome").classList.add("hidden");
  status("running: your picture stays on this device", "good");
  let lastVideoTime = -1;
  const loop = () => {
    if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const now = performance.now();
      onHand(detect(video, now), now / 1000, video.videoHeight / video.videoWidth);
    }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

async function selfTest(model) {
  const fixture = await (await fetch("parity.json")).json();
  const { checks, failures } = AP.checkParity(model, fixture);
  status(`self-test: ${checks - failures.length} of ${checks} checks agree with the Python app`, failures.length ? "bad" : "good");
  $("welcome-note").textContent = failures.length ? failures.slice(0, 3).join(" | ") : "Self-test passed: this page computes what the desktop app computes.";
  return { checks, failures };
}

async function engineTest() {
  await loadEngine();
  const blank = document.createElement("canvas");
  blank.width = 640; blank.height = 360;
  blank.getContext("2d").fillRect(0, 0, 640, 360);
  const seen = detect(blank, performance.now());
  status(`hand tracking loaded: ${seen ? "a hand" : "no hand"} in an empty picture`, seen ? "bad" : "good");
  return seen;
}

function show(page, button) {
  frame.src = `../lab/${page}.html`;
  for (const id of ["go-tutorial", "go-test"]) $(id).classList.toggle("on", id === button);
}

async function main() {
  const model = await (await fetch("model.json")).json();
  pipeline = new AP.Pipeline(model);
  $("start").addEventListener("click", startCamera);
  $("go-tutorial").addEventListener("click", () => show("tutorial", "go-tutorial"));
  $("go-test").addEventListener("click", () => show("test", "go-test"));
  $("dwell").addEventListener("click", () => {
    pipeline.dwellOn = !pipeline.dwellOn;
    pipeline.dwell.reset();
    $("dwell").textContent = `Dwell click: ${pipeline.dwellOn ? "on" : "off"}`;
    $("dwell").classList.toggle("on", pipeline.dwellOn);
  });
  // For checking the page without a camera: recorded hands can be fed in from the console or a test.
  window.AirPointDebug = { AP, pipeline, pointer, onHand, selfTest: () => selfTest(model), engineTest,
                           show, model, frame };
  status("ready");
  if (params.has("selftest")) await selfTest(model);
  if (params.has("engine")) await engineTest();
  if (params.has("replay")) { $("welcome").classList.add("hidden"); status("replay: no camera, recorded hands are fed in by a script"); }
  if (params.has("robot")) {                       // a robot hand does the tutorial or the test: see robot.js
    const what = params.get("robot") === "test" ? "test" : "tutorial";
    $("welcome").classList.add("hidden");
    show(what, `go-${what}`);
    frame.addEventListener("load", async () => {
      const { run } = await import("./robot.js");
      const out = await run(window.AirPointDebug, what);
      status(out.error ? `robot hand: ${out.error}` : `robot hand finished the ${what}`, out.error ? "bad" : "good");
    }, { once: true });
  }
}

main().catch(e => status(`could not load: ${e.message}`, "bad"));
