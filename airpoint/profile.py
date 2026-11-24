"""Everything personal about one user, in profiles/<user>.json.

    venv/bin/python -m airpoint.profile ana                      # show
    venv/bin/python -m airpoint.profile ana --role grip=pinch --role right_click=fist --off scroll_down
    venv/bin/python -m airpoint.profile ana --dwell on --region 0.3,0.3,0.7,0.8
"""
import argparse
import json
import os
import re
from dataclasses import asdict, dataclass, field, fields

from .actions import ACTIVE_REGION
from .fsm import Event
from .pointer import FC_MIN, MODES
from .poses import POSES

VERSION = 2
DEFAULT_REGION = ACTIVE_REGION
ROLE_NAMES = ("grip", "right_click", "scroll_up", "scroll_down")


def default_roles():
    return {"grip": ["pinch", "fist"], "right_click": ["v_sign"], "scroll_up": ["thumb_up"], "scroll_down": ["thumb_down"]}


@dataclass
class Profile:
    user: str
    model: str = None               # personal model and labels, paths relative to the project root
    labels: str = None
    acc_on_own_frames: float = None
    fc_min: float = FC_MIN
    region: list = field(default_factory=lambda: list(DEFAULT_REGION))   # comfort box, normalized frame coords
    roles: dict = field(default_factory=default_roles)
    skipped: list = field(default_factory=list)    # poses the hand could not hold at calibration
    dwell: bool = False
    sounds: bool = True
    hints: bool = True
    filter: str = "full"
    signs: dict = field(default_factory=dict)      # taught signs: name -> {"action", "arg", "hold_s"}
    voice: str = None                              # `say` voice for spoken phrases; None = the system voice
    version: int = VERSION


def path(root, user):
    return os.path.join(root, "profiles", f"{user}.json")


def load(root, user):
    """-> Profile, or None. A Week 2 file (no version) keeps its fields and gains the defaults."""
    p = path(root, user)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        raw = json.load(f)
    known = {f.name for f in fields(Profile)}
    prof = Profile(**{k: v for k, v in raw.items() if k in known and k != "version"})
    prof.roles = {name: list(prof.roles.get(name, [])) for name in ROLE_NAMES}
    return prof


def save(root, profile):
    p = path(root, profile.user)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(asdict(profile), f, indent=2)
    return p


def check_roles(roles):
    """-> what is wrong with a role map, or None."""
    seen = {}
    for name, poses in roles.items():
        if name not in ROLE_NAMES:
            return f"unknown role '{name}' (roles: {', '.join(ROLE_NAMES)})"
        if not isinstance(poses, list) or not all(isinstance(p, str) for p in poses):
            return f"role '{name}' needs a list of pose names"
        for pose in poses:
            if pose in seen:
                return f"pose '{pose}' is used twice: {seen[pose]} and {name}"
            seen[pose] = name
    return None


def drop_poses(roles, poses):
    return {name: [p for p in roles.get(name, []) if p not in poses] for name in ROLE_NAMES}


SIGN_ACTIONS = ("say", "hotkey", "type", "open", "pause")
SIGN_HOLD_S = 0.6        # longer than the built-in 0.3 s: a taught sign must be meant
KEY_ALIASES = {"cmd": "command", "ctrl": "control", "opt": "option", "alt": "option", "return": "enter", "esc": "escape"}
KEY_NAMES = ({"command", "control", "option", "shift", "enter", "escape", "space", "tab", "backspace", "delete",
              "left", "right", "up", "down", "home", "end", "pageup", "pagedown"}
             | {f"f{i}" for i in range(1, 13)})


def check_sign_name(name, signs):
    """-> what is wrong with a name for a taught sign, or None. Re-teaching an existing sign is allowed."""
    if not re.fullmatch(r"[a-z0-9_]{1,20}", name or ""):
        return "use 1-20 lower-case letters, digits or _"
    if name in POSES:
        return f"'{name}' is a built-in pose"
    return None


def parse_keys(text):
    """'cmd+shift+4' -> ('command', 'shift', '4'), or None if a key is not recognised."""
    keys = [KEY_ALIASES.get(k.strip().lower(), k.strip().lower()) for k in text.split("+")]
    ok = all(k in KEY_NAMES or (len(k) == 1 and k.isprintable() and k != " ") for k in keys)
    return tuple(keys) if text.strip() and ok else None


def fsm_parts(roles, signs=None):
    """Profile roles and taught signs -> (grip set, scroll map, hold events, hold seconds) for AirPointFSM."""
    scroll = {p: 1 for p in roles.get("scroll_up", [])}
    scroll.update({p: -1 for p in roles.get("scroll_down", [])})
    hold = {p: Event("right_click") for p in roles.get("right_click", [])}
    hold_s = {}
    for name, spec in (signs or {}).items():
        arg = tuple(spec["arg"]) if spec["action"] == "hotkey" else spec.get("arg")
        hold[name] = Event(spec["action"], arg)
        hold_s[name] = spec.get("hold_s", SIGN_HOLD_S)
    return frozenset(roles.get("grip", [])), scroll, hold, hold_s


def _on_off(text):
    if text not in ("on", "off"):
        raise argparse.ArgumentTypeError("use on or off")
    return text == "on"


def main(argv=None, root=None):
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description="Show or edit one user's AirPoint profile")
    ap.add_argument("user")
    ap.add_argument("--role", action="append", default=[], metavar="NAME=pose[,pose]",
                    help=f"which poses do what; roles: {', '.join(ROLE_NAMES)}")
    ap.add_argument("--off", action="append", default=[], metavar="NAME", help="give a role no pose")
    ap.add_argument("--reset-roles", action="store_true")
    ap.add_argument("--region", metavar="x0,y0,x1,y1", help="comfort box in frame fractions")
    ap.add_argument("--reset-region", action="store_true")
    ap.add_argument("--dwell", type=_on_off, metavar="on|off")
    ap.add_argument("--sounds", type=_on_off, metavar="on|off")
    ap.add_argument("--hints", type=_on_off, metavar="on|off")
    ap.add_argument("--filter", choices=MODES)
    ap.add_argument("--voice", help="voice for spoken phrases, e.g. Rishi, Tara, Lekha (see: say -v '?')")
    args = ap.parse_args(argv)

    prof = load(root, args.user) or Profile(user=args.user)
    before = asdict(prof)
    if args.reset_roles:
        prof.roles = default_roles()
    for item in args.role:
        name, _, poses = item.partition("=")
        prof.roles = drop_poses(prof.roles, []) | {name: [p for p in poses.split(",") if p]}
    for name in args.off:
        prof.roles = prof.roles | {name: []}
    problem = check_roles(prof.roles)
    if problem:
        raise SystemExit(problem)
    if args.reset_region:
        prof.region = list(DEFAULT_REGION)
    if args.region:
        box = [float(v) for v in args.region.split(",")]
        if len(box) != 4 or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
            raise SystemExit("region needs x0,y0,x1,y1 with 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
        prof.region = box
    for name in ("dwell", "sounds", "hints", "filter", "voice"):
        if getattr(args, name) is not None:
            setattr(prof, name, getattr(args, name))
    if asdict(prof) != before or not os.path.exists(path(root, args.user)):
        if asdict(prof) != before:
            save(root, prof)
    print(json.dumps(asdict(prof), indent=2))


if __name__ == "__main__":
    main()
