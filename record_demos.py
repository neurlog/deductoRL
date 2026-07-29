"""
Record human demonstrations for behavior-cloning pretraining.

Uses the same capture and step pacing as env.py, so recorded observations are
identical in format to what the agent will see. Hold W the entire time. The
agent holds forward automatically, so demos must match its action semantics.

    python record_demos.py --check-input          # verify input is seen FIRST
    python record_demos.py --out demos/run --episodes 40 --keep-falls
"""

import argparse
import os
import sys
import threading
import time

import numpy as np

import config
from capture import ScreenCapture
from hotkey import is_stop_hotkey_pressed
from input_controller import is_key_held
from reward import TimerTracker, ocr_timer_seconds

_IS_MAC = sys.platform == "darwin"


class InputListener:
    """Accumulates real mouse deltas and SPACE presses between steps.

    Polling can miss a quick jump tap, and once a game captures the cursor
    there is no polling API for relative mouse motion at all. Hence a hook.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._dx = self._dy = 0.0
        self._space = False
        self.events_seen = 0
        self._stop = None

    def _add(self, dx, dy, space):
        with self._lock:
            self._dx += dx
            self._dy += dy
            self._space = self._space or space
            self.events_seen += 1

    def start(self):
        if _IS_MAC:
            self._start_quartz()
        else:
            self._start_pynput()
        time.sleep(0.5)

    def _start_quartz(self):
        import Quartz

        def cb(proxy, etype, event, refcon):
            if etype in (Quartz.kCGEventMouseMoved,
                         Quartz.kCGEventLeftMouseDragged,
                         Quartz.kCGEventRightMouseDragged):
                self._add(
                    Quartz.CGEventGetIntegerValueField(event, Quartz.kCGMouseEventDeltaX),
                    Quartz.CGEventGetIntegerValueField(event, Quartz.kCGMouseEventDeltaY),
                    False)
            elif etype == Quartz.kCGEventKeyDown:
                if Quartz.CGEventGetIntegerValueField(
                        event, Quartz.kCGKeyboardEventKeycode) == 49:  # SPACE
                    self._add(0, 0, True)
            return event

        def run():
            mask = (Quartz.CGEventMaskBit(Quartz.kCGEventMouseMoved)
                    | Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseDragged)
                    | Quartz.CGEventMaskBit(Quartz.kCGEventRightMouseDragged)
                    | Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown))
            tap = Quartz.CGEventTapCreate(
                Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap,
                Quartz.kCGEventTapOptionListenOnly, mask, cb, None)
            if tap is None:
                print("[recorder] FAILED to create event tap. Grant Accessibility "
                      "permission to whatever runs this script.")
                return
            loop = Quartz.CFRunLoopGetCurrent()
            Quartz.CFRunLoopAddSource(
                loop, Quartz.CFMachPortCreateRunLoopSource(None, tap, 0),
                Quartz.kCFRunLoopCommonModes)
            Quartz.CGEventTapEnable(tap, True)
            self._stop = lambda: Quartz.CFRunLoopStop(loop)
            Quartz.CFRunLoopRun()

        threading.Thread(target=run, daemon=True).start()

    def _start_pynput(self):
        from pynput import keyboard, mouse

        last = {"pos": None}

        def on_move(x, y):
            if last["pos"] is not None:
                self._add(x - last["pos"][0], y - last["pos"][1], False)
            last["pos"] = (x, y)

        def on_press(key):
            if key == keyboard.Key.space:
                self._add(0, 0, True)

        ml = mouse.Listener(on_move=on_move)
        kl = keyboard.Listener(on_press=on_press)
        ml.start()
        kl.start()
        self._stop = lambda: (ml.stop(), kl.stop())

    def stop(self):
        if self._stop:
            self._stop()

    def consume(self):
        with self._lock:
            out = (self._dx, self._dy, self._space)
            self._dx = self._dy = 0.0
            self._space = False
        return out


def nearest_bin(value, bins):
    return int(np.argmin([abs(value - b) for b in bins]))


def check_input(listener):
    print("Move the mouse and tap SPACE inside the game window. Ctrl+C to quit.\n")
    try:
        while True:
            time.sleep(config.STEP_DT)
            dx, dy, space = listener.consume()
            print(f"  dx={dx:+7.0f} -> bin {nearest_bin(dx, config.LOOK_DELTAS_X):2d}"
                  f"   dy={dy:+6.0f} -> bin {nearest_bin(dy, config.LOOK_DELTAS_Y)}"
                  f"   jump={int(space)}   W={int(is_key_held('W'))}"
                  f"   events={listener.events_seen}", end="\r")
    except KeyboardInterrupt:
        print("\n\nIf dx/dy stayed 0 while you moved the mouse, the hook isn't "
              "receiving events. Check input permissions.")


def record_episode(capture, listener, max_seconds):
    """Record until a fall/finish is detected or time runs out.

    Pre-timer frames ARE recorded: reaching the first platform is part of the
    task, and the timer only starts once you land on it.
    """
    obs_list, act_list = [], []
    w_held = 0
    armed_index = -1
    start = time.time()
    tracker = TimerTracker(start_time=start)
    tracker.update(capture.crop_timer(capture.grab_bgr()))
    outcome, final_time = "timeout", None
    listener.consume()

    while True:
        deadline = time.time() + config.STEP_DT
        frame = capture.grab_bgr()
        obs = capture.to_observation(frame)
        timer_crop = capture.crop_timer(frame)

        dx, dy, space = listener.consume()
        obs_list.append(obs)
        act_list.append((int(space),
                         nearest_bin(dx, config.LOOK_DELTAS_X),
                         nearest_bin(dy, config.LOOK_DELTAS_Y)))
        if is_key_held("W"):
            w_held += 1

        was_armed = tracker.armed
        event = tracker.update(timer_crop)
        if tracker.armed and not was_armed:
            armed_index = len(obs_list) - 1

        if event == "finish":
            outcome = "finish"
            final_time = ocr_timer_seconds(timer_crop) or (time.time() - start)
            break
        if event in ("fell", "never_started"):
            outcome = event
            break
        if time.time() - start > max_seconds:
            break
        if is_stop_hotkey_pressed():
            outcome = "aborted"
            break

        remaining = deadline - time.time()
        if remaining > 0:
            time.sleep(remaining)

    return (np.asarray(obs_list, dtype=np.uint8),
            np.asarray(act_list, dtype=np.int64),
            outcome, final_time, w_held / max(len(obs_list), 1), armed_index)


def main():
    ap = argparse.ArgumentParser(description="Record demos for BC pretraining.")
    ap.add_argument("--out", default="demos/run")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--countdown", type=float, default=3.0)
    ap.add_argument("--max-episode-seconds", type=float, default=None)
    ap.add_argument("--keep-falls", action="store_true",
                    help="Also save falls (teaches the value head what precedes a fall).")
    ap.add_argument("--check-input", action="store_true",
                    help="Print live input readings and exit. Do this first.")
    args = ap.parse_args()

    listener = InputListener()
    listener.start()

    if args.check_input:
        check_input(listener)
        listener.stop()
        return

    max_seconds = args.max_episode_seconds or config.MAX_EPISODE_SECONDS
    os.makedirs(args.out, exist_ok=True)
    capture = ScreenCapture()

    print(f"Recording up to {args.episodes} episodes into {args.out}/")
    print("Hold W throughout. Press your respawn key during each countdown so "
          "every demo starts where the agent will.\nShift+Alt+K to stop early.\n")

    kept = 0
    try:
        for ep in range(args.episodes):
            print(f"[episode {ep + 1}/{args.episodes}] get ready...")
            for i in range(int(args.countdown), 0, -1):
                print(f"  starting in {i}...", end="\r")
                time.sleep(1.0)
            print("  GO" + " " * 20)

            obs, acts, outcome, final_time, w_frac, armed_index = record_episode(
                capture, listener, max_seconds)
            jump_rate = float(acts[:, 0].mean()) if len(acts) else 0.0
            print(f"  -> {outcome}, {len(obs)} steps, jump rate {jump_rate:.1%}, "
                  f"W held {w_frac:.0%}"
                  + (f", time {final_time:.2f}s" if final_time else ""))

            if w_frac < 0.9:
                print("     W wasn't held for most of the episode. This "
                      "demo doesn't match the agent's action semantics.")
            if outcome == "aborted":
                break
            if outcome != "finish" and not args.keep_falls:
                print("  not a finish. Discarded.")
                continue

            path = os.path.join(args.out, f"demo_{int(time.time())}_{ep:03d}.npz")
            np.savez_compressed(
                path, obs=obs, actions=acts, outcome=outcome,
                final_time=(final_time if final_time is not None else -1.0),
                armed_index=armed_index, step_dt=config.STEP_DT,
                look_deltas_x=np.array(config.LOOK_DELTAS_X),
                look_deltas_y=np.array(config.LOOK_DELTAS_Y))
            kept += 1
            print(f"  saved {path}")

            if is_stop_hotkey_pressed():
                break
    finally:
        listener.stop()
        capture.close()
        print(f"\nDone. {kept} episodes saved in {args.out}/")


if __name__ == "__main__":
    main()
