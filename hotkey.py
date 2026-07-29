"""Shift+Option/Alt+K. Stops training or recording cleanly."""

import sys

if sys.platform == "darwin":
    import Quartz

    _K, _SHIFT, _ALT = (40,), (56, 60), (58, 61)

    def _down(codes):
        return any(Quartz.CGEventSourceKeyState(
            Quartz.kCGEventSourceStateHIDSystemState, c) for c in codes)
else:
    import ctypes

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _K, _SHIFT, _ALT = (0x4B,), (0x10,), (0x12,)

    def _down(codes):
        return any(_user32.GetAsyncKeyState(c) & 0x8000 for c in codes)


def is_stop_hotkey_pressed() -> bool:
    return _down(_SHIFT) and _down(_ALT) and _down(_K)
