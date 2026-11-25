"""A picture of the screen for the AI helper, with AirPoint's own windows blacked out. macOS."""
import os
import subprocess
import tempfile
from dataclasses import dataclass

import cv2

MIN_WINDOW_PX = 40      # smaller windows of ours (tooltips, status items) are not worth a black box


@dataclass
class Shot:
    jpeg: bytes
    app: str            # the app in front; "" when it cannot be told
    size: tuple         # (w, h) of the picture: the screen in points


def allowed():
    """May this process record the screen? None when the system cannot say."""
    try:
        from Quartz import CGPreflightScreenCaptureAccess
        return bool(CGPreflightScreenCaptureAccess())
    except Exception:
        return None


def front_app():
    try:
        from AppKit import NSWorkspace
        return str(NSWorkspace.sharedWorkspace().frontmostApplication().localizedName() or "")
    except Exception:
        return ""


def own_windows(pids):
    """Windows on screen that belong to these processes -> [(x, y, w, h)] in screen points."""
    try:
        from Quartz import CGWindowListCopyWindowInfo, kCGNullWindowID, kCGWindowListOptionOnScreenOnly
        boxes = []
        for w in CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID) or []:
            b = w.get("kCGWindowBounds") or {}
            if w.get("kCGWindowOwnerPID") in pids and min(b.get("Width", 0), b.get("Height", 0)) >= MIN_WINDOW_PX:
                boxes.append((float(b["X"]), float(b["Y"]), float(b["Width"]), float(b["Height"])))
        return boxes
    except Exception:
        return []


def capture(run=subprocess.run):
    """The main display as a BGR image, in pixels. The file it passes through is deleted at once."""
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        run(["screencapture", "-x", "-m", "-t", "png", path], check=True, timeout=10,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        img = cv2.imread(path)
    except (OSError, subprocess.SubprocessError) as e:
        raise OSError(f"no picture of the screen ({type(e).__name__})") from None
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    if img is None:
        raise OSError("no picture of the screen: allow Screen Recording for your terminal in System Settings")
    return img


def shot(screen_size, hide=(), capture=capture, front=front_app, quality=80):
    """-> Shot: the screen scaled to its size in points, the `hide` boxes (points) blacked out, as JPEG."""
    img = capture()
    w, h = int(screen_size[0]), int(screen_size[1])
    if (img.shape[1], img.shape[0]) != (w, h):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    for x, y, bw, bh in hide:
        x0, y0 = max(int(x), 0), max(int(y), 0)
        img[y0:max(min(int(y + bh) + 1, h), y0), x0:max(min(int(x + bw) + 1, w), x0)] = 0
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise OSError("the picture of the screen could not be packed")
    return Shot(buf.tobytes(), front(), (w, h))
