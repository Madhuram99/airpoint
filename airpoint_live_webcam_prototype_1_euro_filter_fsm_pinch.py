"""
AirPoint Live Webcam Prototype — Midterm Evaluation Showcase
Department of Computer Science & Engineering, PEC Chandigarh
Authors: Madhuram Rathi, Ishan Gupta, Dakshmeet Singh, Bhuvi Kansal

Run this script during your presentation to demonstrate:
1. Real-time 21-point 3D hand tracking via MediaPipe Hands (Zhang et al., 2020)
2. Live visual comparison:
   - RED circle   = Raw shaky palm coordinates
   - GREEN circle = 1-Euro Filtered smooth cursor (Casiez et al., 2012)
3. Scale-free pinch ratio with Hysteresis (`enter < 0.28`, `exit > 0.36`)
4. Slide-6 Finite State Machine (`RESTING`, `MOVING`, `PRESSING`, `DRAGGING`, `SCROLLING`)
5. Press 'm' to toggle real OS mouse control (pinch = click, hold pinch = drag,
   thumb pose + move hand up/down = scroll). Press 'q' to quit.
"""

import cv2
import math
import time
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# OS mouse control starts OFF (safe mode) — press 'm' in the video window to toggle it.
# On macOS, allow your terminal app in System Settings > Privacy & Security > Accessibility.
try:
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

# Inner part of the camera frame that maps to the full screen, so you can
# reach the screen edges without your hand leaving the camera view.
ACTIVE_REGION = (0.15, 0.15, 0.85, 0.80)  # x_min, y_min, x_max, y_max (normalized)
SCROLL_GAIN = 0.6                          # scroll clicks per pixel of vertical hand motion


# ==============================================================================
# 1. ONE-EURO FILTER (Slide 7)
# ==============================================================================
class OneEuroFilter:
    def __init__(self, fc_min=0.80, beta=0.010, d_cutoff=1.0):
        self.fc_min = fc_min
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(fc, dt):
        return 1.0 / (1.0 + 1.0 / (2.0 * math.pi * fc * dt))

    def filter(self, x, t, custom_fc_min=None):
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x

        dt = max(t - self.t_prev, 1e-5)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        edx = a_d * dx + (1.0 - a_d) * self.dx_prev

        base_fc = custom_fc_min if custom_fc_min is not None else self.fc_min
        fc = base_fc + self.beta * abs(edx)
        alpha = self._alpha(fc, dt)
        x_out = alpha * x + (1.0 - alpha) * self.x_prev

        self.x_prev, self.dx_prev, self.t_prev = x_out, edx, t
        return x_out


# ==============================================================================
# 2. SCALE-FREE GEOMETRY & SLIDE-6 FINITE STATE MACHINE
# ==============================================================================
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_MCP, MIDDLE_TIP = 9, 12
RING_TIP = 16
PINKY_TIP = 20


def dist_2d(p1, p2):
    """Euclidean distance between two landmark objects (with .x and .y attrs)."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


class AirPointFSM:
    """
    Implements the exact Finite State Machine from Slide 6:
    States: RESTING, MOVING, PRESSING, DRAGGING, SCROLLING
    """
    def __init__(self):
        self.state = "RESTING"
        self.pinch_active = False
        self.press_start_time = 0.0
        self.last_event = "None"
        self.action = None  # one-shot OS action for this frame: "click" | "drag_start" | "drop"

    def update(self, landmarks, t_now):
        self.action = None
        if landmarks is None:
            if self.state == "DRAGGING":
                self.action = "drop"  # never leave the OS mouse button stuck down
            self.state = "RESTING"
            self.pinch_active = False
            return self.state, 1.0, self.last_event

        palm_size = dist_2d(landmarks[WRIST], landmarks[MIDDLE_MCP]) + 1e-6
        pinch_ratio = dist_2d(landmarks[THUMB_TIP], landmarks[INDEX_TIP]) / palm_size

        # Hysteresis pinch check (Enter < 0.28, Release > 0.36)
        if self.pinch_active:
            self.pinch_active = (pinch_ratio < 0.36)
        else:
            self.pinch_active = (pinch_ratio < 0.28)

        # Check thumb scroll posture (Thumb extended vertically, other fingers curled)
        idx_ext = dist_2d(landmarks[INDEX_TIP], landmarks[WRIST]) / palm_size
        mid_ext = dist_2d(landmarks[MIDDLE_TIP], landmarks[WRIST]) / palm_size
        thumb_ext = dist_2d(landmarks[THUMB_TIP], landmarks[WRIST]) / palm_size
        is_thumb_scroll = (thumb_ext > 1.35 and idx_ext < 0.98 and mid_ext < 0.98)

        if self.state == "RESTING":
            self.state = "MOVING"

        elif self.state == "MOVING":
            if self.pinch_active:
                self.state = "PRESSING"
                self.press_start_time = t_now
            elif is_thumb_scroll:
                self.state = "SCROLLING"
                self.last_event = "SCROLL ACTIVE"

        elif self.state == "PRESSING":
            held_time = t_now - self.press_start_time
            if not self.pinch_active:
                if held_time < 0.50:
                    self.last_event = f"LEFT CLICK! ({held_time*1000:.0f} ms)"
                    self.action = "click"
                self.state = "MOVING"
            elif held_time >= 0.50:
                self.state = "DRAGGING"
                self.last_event = "DRAG HOLD ENGAGED"
                self.action = "drag_start"

        elif self.state == "DRAGGING":
            if not self.pinch_active:
                self.last_event = "DROP (DRAG RELEASED)"
                self.state = "MOVING"
                self.action = "drop"

        elif self.state == "SCROLLING":
            if not is_thumb_scroll:
                self.state = "MOVING"

        return self.state, pinch_ratio, self.last_event


class OSMouse:
    """Maps the filtered cursor and FSM actions onto the real OS mouse via pyautogui."""
    def __init__(self):
        self.enabled = False
        self.button_down = False
        self.prev_scroll_y = None
        self.screen_w, self.screen_h = pyautogui.size() if PYAUTOGUI_AVAILABLE else (0, 0)

    def toggle(self):
        if not PYAUTOGUI_AVAILABLE:
            print("[MOUSE] pyautogui not installed — run: pip install pyautogui")
            return
        self.release()
        self.enabled = not self.enabled
        print(f"[MOUSE] OS mouse control {'ON' if self.enabled else 'OFF'}")

    def to_screen(self, nx, ny):
        x0, y0, x1, y1 = ACTIVE_REGION
        sx = min(max((nx - x0) / (x1 - x0), 0.0), 1.0) * (self.screen_w - 1)
        sy = min(max((ny - y0) / (y1 - y0), 0.0), 1.0) * (self.screen_h - 1)
        return sx, sy

    def apply(self, nx, ny, state, action, frame_y):
        if not self.enabled:
            return
        if state == "SCROLLING":
            # Thumb pose: vertical hand motion scrolls; cursor stays put
            if self.prev_scroll_y is not None:
                clicks = int((self.prev_scroll_y - frame_y) * SCROLL_GAIN)
                if clicks:
                    pyautogui.scroll(clicks)
                    self.prev_scroll_y = frame_y
            else:
                self.prev_scroll_y = frame_y
            return
        self.prev_scroll_y = None

        pyautogui.moveTo(*self.to_screen(nx, ny))
        if action == "click":
            pyautogui.click()
        elif action == "drag_start":
            pyautogui.mouseDown()
            self.button_down = True
        elif action == "drop":
            self.release()

    def release(self):
        if self.button_down:
            pyautogui.mouseUp()
            self.button_down = False
        self.prev_scroll_y = None


# ==============================================================================
# 3. LIVE WEBCAM LOOP (MediaPipe Tasks API — v1.x)
# ==============================================================================
def run_live_demo():
    # Setup MediaPipe HandLandmarker with the new Tasks API
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at '{model_path}'")
        print("       Download it with:")
        print('       curl -L -o hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"')
        return

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Hand connections for drawing (same as MediaPipe HAND_CONNECTIONS)
    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),        # Thumb
        (0,5),(5,6),(6,7),(7,8),        # Index
        (0,9),(9,10),(10,11),(11,12),   # Middle
        (0,13),(13,14),(14,15),(15,16), # Ring
        (0,17),(17,18),(18,19),(19,20), # Pinky
        (5,9),(9,13),(13,17)            # Palm
    ]

    # Index 1 = MacBook built-in camera when an iPhone Continuity Camera takes index 0;
    # with no iPhone nearby the built-in camera is index 0, so fall back to it.
    cap = None
    for cam_index in (1, 0):
        cap = cv2.VideoCapture(cam_index)
        if cap.isOpened() and cap.read()[0]:
            print(f"[LIVE DEMO] Using camera index {cam_index}.")
            break
        cap.release()
    else:
        print("[ERROR] Cannot open webcam. Check camera permissions in System Settings > Privacy > Camera.")
        return

    filter_x = OneEuroFilter(fc_min=0.80, beta=0.010)
    filter_y = OneEuroFilter(fc_min=0.80, beta=0.010)
    fsm = AirPointFSM()
    mouse = OSMouse()
    prev_frame_time = time.time()

    print("[LIVE DEMO] Webcam started. Press 'm' to toggle OS mouse control, 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        t_now = time.time()
        fps = 1.0 / max(t_now - prev_frame_time, 1e-5)
        prev_frame_time = t_now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert frame to MediaPipe Image and detect
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = detector.detect(mp_image)

        if results.hand_landmarks and len(results.hand_landmarks) > 0:
            hand_lms = results.hand_landmarks[0]  # list of NormalizedLandmark

            # Draw hand skeleton using pure OpenCV
            for (a, b) in HAND_CONNECTIONS:
                x1, y1 = int(hand_lms[a].x * w), int(hand_lms[a].y * h)
                x2, y2 = int(hand_lms[b].x * w), int(hand_lms[b].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (250, 44, 250), 2)
            for i, lm in enumerate(hand_lms):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (121, 22, 76), -1)

            # FSM update — hand_lms is a list of landmarks with .x, .y, .z
            state, pinch_ratio, last_event = fsm.update(hand_lms, t_now)

            # Anchor cursor to MIDDLE_MCP (Landmark 9) so pinching doesn't shift the cursor
            raw_x = int(hand_lms[MIDDLE_MCP].x * w)
            raw_y = int(hand_lms[MIDDLE_MCP].y * h)

            # Pre-click stabilization: lower fc_min to 0.30 Hz when fingers start closing
            active_fc_min = 0.30 if pinch_ratio < 0.42 else 0.80
            smooth_x = int(filter_x.filter(raw_x, t_now, custom_fc_min=active_fc_min))
            smooth_y = int(filter_y.filter(raw_y, t_now, custom_fc_min=active_fc_min))

            # Draw Raw (Red = Shaky) vs 1-Euro Filtered (Green = Smooth)
            cv2.circle(frame, (raw_x, raw_y), 6, (0, 0, 255), -1)
            cv2.circle(frame, (smooth_x, smooth_y), 12, (0, 255, 0), 2)

            mouse.apply(smooth_x / w, smooth_y / h, state, fsm.action, smooth_y)

        else:
            state, pinch_ratio, last_event = fsm.update(None, t_now)
            if fsm.action == "drop":
                mouse.release()

        # Active region that maps onto the full screen
        x0, y0, x1, y1 = ACTIVE_REGION
        cv2.rectangle(frame, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)),
                      (0, 200, 255) if mouse.enabled else (90, 90, 90), 1)

        # HUD Telemetry Banner
        cv2.rectangle(frame, (0, 0), (w, 82), (25, 35, 30), -1)
        cv2.putText(
            frame,
            f"AirPoint Prototype | State: {state} | Pinch Ratio: {pinch_ratio:.2f} | FPS: {fps:.1f}",
            (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (167, 243, 208), 2
        )
        cv2.putText(
            frame,
            f"Last Event: {last_event}  |  Red: Raw  |  Green: 1-Euro  |  "
            f"OS Mouse [m]: {'ON' if mouse.enabled else 'OFF'}",
            (15, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
            (0, 200, 255) if mouse.enabled else (255, 255, 255), 1
        )

        cv2.imshow("AirPoint Midterm Live Showcase (PEC Chandigarh)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            mouse.toggle()

    mouse.release()
    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    run_live_demo()