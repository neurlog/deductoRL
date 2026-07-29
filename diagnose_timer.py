"""Calibrate timer detection. Run it, play a full run, and it prints measured
statistics plus recommended config values.

    python diagnose_timer.py
    python diagnose_timer.py --save-crops out/    # dump ROI images to inspect
"""

import argparse
import os
import time

import cv2
import numpy as np

import config
from capture import ScreenCapture
from reward import TimerTracker, diff_pixels, threshold_timer, white_pixels


def main():
    parser = argparse.ArgumentParser(description="Diagnose/calibrate timer detection.")
    parser.add_argument("--save-crops", default=None,
                        help="Directory to dump raw + thresholded ROI crops into.")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save every Nth frame when --save-crops is set.")
    args = parser.parse_args()

    if args.save_crops:
        os.makedirs(args.save_crops, exist_ok=True)

    capture = ScreenCapture()
    tracker = TimerTracker()

    whites, diffs_prearm, diffs_armed = [], [], []
    prev_thresh = None
    frame_i = 0
    events = []

    print("Watching the timer ROI. Play a full run (start it, finish it).")
    print("Ctrl+C when done.\n")
    print(f"  current config: PRESENT_MIN={config.TIMER_PRESENT_MIN_PIXELS}  "
          f"CHANGED_MIN={config.TIMER_CHANGED_MIN_PIXELS}  "
          f"FREEZE_SECONDS={config.FINISH_FREEZE_SECONDS}\n")

    try:
        while True:
            deadline = time.time() + config.STEP_DT

            frame = capture.grab_bgr()
            crop = capture.crop_timer(frame)
            thresh = threshold_timer(crop)

            w = white_pixels(thresh)
            d = diff_pixels(prev_thresh, thresh)
            prev_thresh = thresh
            whites.append(w)
            if d >= 0:
                (diffs_armed if tracker.armed else diffs_prearm).append(d)

            was_armed = tracker.armed
            event = tracker.update(crop)
            if tracker.armed and not was_armed:
                events.append("ARMED")
                print("\n  *** ARMED. Timer started ***")

            if args.save_crops and frame_i % args.save_every == 0:
                cv2.imwrite(os.path.join(args.save_crops, f"{frame_i:05d}_raw.png"), crop)
                cv2.imwrite(os.path.join(args.save_crops, f"{frame_i:05d}_thresh.png"), thresh)

            state = "ARMED " if tracker.armed else "pre-run"
            print(f"  {state}  white={w:5d}  diff={d:5d}  present={int(w >= config.TIMER_PRESENT_MIN_PIXELS)}"
                  f"  freeze={tracker.freeze_seconds:4.1f}s   ", end="\r")

            if event:
                print(f"\n  *** EVENT: {event} ***  (tracker resets, keep playing)")
                events.append(event)
                tracker = TimerTracker()
                prev_thresh = None

            frame_i += 1
            remaining = deadline - time.time()
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        capture.close()

    # ---------------- summary + recommendations ----------------
    print("\n\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)
    if not whites:
        print("No frames captured.")
        return

    whites = np.array(whites)
    print(f"frames: {frame_i}   events: {events or 'none'}")
    print(f"white pixels  min={whites.min()}  median={int(np.median(whites))}  max={whites.max()}")

    if diffs_armed:
        da = np.array(diffs_armed)
        # The noise floor is the low end of the diff distribution: frames
        # where the timer did NOT tick. The signal is the high end.
        noise = int(np.percentile(da, 25))
        signal = int(np.percentile(da, 75))
        print(f"diff (armed)  p25(noise)={noise}  median={int(np.median(da))}  p75(signal)={signal}  max={da.max()}")
    else:
        noise = signal = None
        print("diff (armed)  no armed frames. The timer never started, or TIMER_ROI is wrong.")

    if diffs_prearm:
        dp = np.array(diffs_prearm)
        print(f"diff (pre-run) median={int(np.median(dp))}  max={dp.max()}")

    print("\nRECOMMENDED config.py values:")
    lo = int(whites.min())
    hi = int(whites.max())
    if hi > lo * 2:
        rec_present = max(5, int(lo + (hi - lo) * 0.25))
        print(f"  TIMER_PRESENT_MIN_PIXELS = {rec_present}"
              f"   (between empty={lo} and full={hi})")
    else:
        print(f"  TIMER_PRESENT_MIN_PIXELS = {max(5, int(hi * 0.25))}"
              f"   (white count never varied much. Did the timer ever"
              f" disappear? If not, this is a guess.)")

    if noise is not None and signal is not None:
        if signal > noise:
            rec_changed = max(4, int((noise + signal) / 2))
            print(f"  TIMER_CHANGED_MIN_PIXELS = {rec_changed}"
                  f"   (above noise={noise}, below signal={signal})")
        else:
            print(f"  TIMER_CHANGED_MIN_PIXELS = ?. Noise and signal overlap"
                  f" (noise={noise}, signal={signal}).")
            print("    Your ROI probably includes changing scenery. Tighten "
                  "TIMER_ROI to just the digits.")

    print("\nIf no FINISH event appeared after you finished a run:")
    print("  - raise TIMER_CHANGED_MIN_PIXELS above the noise figure above")
    print("  - or raise FINISH_FREEZE_SECONDS if your timer ticks slowly")


if __name__ == "__main__":
    main()
