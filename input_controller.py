"""Simulated keyboard/mouse input. macOS uses Quartz, Windows uses SendInput.

Windows games usually read DirectInput, which ignores virtual-key events, so
keys are sent as hardware SCAN codes and the mouse as relative deltas.
"""

import sys
import time

import config

_IS_MAC = sys.platform == "darwin"
_IS_WIN = sys.platform in ("win32", "cygwin")

if _IS_MAC:
    import subprocess

    import Quartz

    _KEYCODES = {"W": 13, "A": 0, "S": 1, "D": 2, "F": 3, "SPACE": 49}
    _source = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)

    def _key(code, down):
        Quartz.CGEventPost(
            Quartz.kCGHIDEventTap,
            Quartz.CGEventCreateKeyboardEvent(_source, code, down),
        )

    def _mouse(dx, dy):
        cx = config.CAPTURE_REGION["left"] + config.CAPTURE_REGION["width"] / 2
        cy = config.CAPTURE_REGION["top"] + config.CAPTURE_REGION["height"] / 2
        ev = Quartz.CGEventCreateMouseEvent(
            _source, Quartz.kCGEventMouseMoved, (cx, cy), Quartz.kCGMouseButtonLeft
        )
        Quartz.CGEventSetIntegerValueField(ev, Quartz.kCGMouseEventDeltaX, dx)
        Quartz.CGEventSetIntegerValueField(ev, Quartz.kCGMouseEventDeltaY, dy)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)

    def activate_app(name=None):
        """Focus the game. CrossOver apps aren't real macOS apps, so target the
        process by the name shown in Activity Monitor."""
        name = name or config.APP_NAME
        script = (f'tell application "System Events" to tell process "{name}" '
                  f'to set frontmost to true')
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[input] couldn't focus '{name}': {r.stderr.strip()}")

elif _IS_WIN:
    import ctypes
    from ctypes import wintypes

    _KEYCODES = {"W": 0x11, "A": 0x1E, "S": 0x1F, "D": 0x20, "F": 0x21, "SPACE": 0x39}
    _ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong
    _user32 = ctypes.WinDLL("user32", use_last_error=True)

    class _KEYBD(ctypes.Structure):
        _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                    ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                    ("dwExtraInfo", _ULONG_PTR)]

    class _MOUSE(ctypes.Structure):
        _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                    ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD), ("dwExtraInfo", _ULONG_PTR)]

    class _U(ctypes.Union):
        _fields_ = [("ki", _KEYBD), ("mi", _MOUSE)]

    class _INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("u", _U)]

    def _send(inp):
        _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))

    def _key(code, down):
        flags = 0x0008 | (0 if down else 0x0002)   # SCANCODE | KEYUP
        _send(_INPUT(type=1, u=_U(ki=_KEYBD(0, code, flags, 0, None))))

    def _mouse(dx, dy):
        _send(_INPUT(type=0, u=_U(mi=_MOUSE(int(dx), int(dy), 0, 0x0001, 0, None))))

    def activate_app(name=None):
        """Focus the first top-level window whose title contains APP_NAME."""
        name = (name or config.APP_NAME).lower().replace(".exe", "")
        found = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def cb(hwnd, _):
            n = _user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                _user32.GetWindowTextW(hwnd, buf, n + 1)
                if name in buf.value.lower():
                    found.append(hwnd)
                    return False
            return True

        _user32.EnumWindows(cb, 0)
        if found:
            _user32.SetForegroundWindow(found[0])
        else:
            print(f"[input] no window titled like '{name}' — is the game running?")

else:
    raise RuntimeError(f"Unsupported platform: {sys.platform}")


def key_down(name):
    _key(_KEYCODES[name], True)


def key_up(name):
    _key(_KEYCODES[name], False)


def release_all_movement_keys():
    for name in config.KEYS:
        key_up(name)


def move_mouse_smooth(dx, dy):
    """Deliver a camera delta as several small events, like a real mouse.
    One large event rotates the view within a single game frame, which is
    motion no human demonstration ever contains."""
    n = max(1, -(-max(abs(dx), abs(dy)) // config.MOUSE_MAX_DELTA_PER_EVENT))
    px = py = 0
    for i in range(1, n + 1):
        cx, cy = round(dx * i / n), round(dy * i / n)
        _mouse(cx - px, cy - py)
        px, py = cx, cy
        if i < n:
            time.sleep(config.MOUSE_EVENT_INTERVAL_SECONDS)


def begin_action(jump_idx, look_h_idx, look_v_idx):
    """Fire one step's input. Returns the horizontal delta."""
    if jump_idx == 1:
        key_down("SPACE")
    dx = config.LOOK_DELTAS_X[look_h_idx]
    dy = config.LOOK_DELTAS_Y[look_v_idx]
    if dx or dy:
        move_mouse_smooth(dx, dy)
    return dx


def end_jump():
    key_up("SPACE")


def is_key_held(name):
    """Physical key state — used by the demo recorder to check W is held."""
    if _IS_MAC:
        return bool(Quartz.CGEventSourceKeyState(
            Quartz.kCGEventSourceStateHIDSystemState, _KEYCODES[name]))
    vk = {"W": 0x57, "A": 0x41, "S": 0x53, "D": 0x44, "F": 0x46, "SPACE": 0x20}[name]
    return bool(_user32.GetAsyncKeyState(vk) & 0x8000)
