import sys
import time

import mss
import Quartz


def live_mouse_position():
    print("Move your mouse over key points (Ctrl+C to stop)...")
    try:
        while True:
            event = Quartz.CGEventCreate(None)
            point = Quartz.CGEventGetLocation(event)
            print(f"\rMouse position: ({point.x:.0f}, {point.y:.0f})", end="", flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nDone.")


def full_screenshot(outfile="calibration_full.png"):
    with mss.mss() as sct:
        sct.shot(output=outfile)
    print(f"Saved {outfile}")


def region_crop(left, top, width, height, outfile):
    region = {"left": left, "top": top, "width": width, "height": height}
    with mss.mss() as sct:
        shot = sct.grab(region)
        mss.tools.to_png(shot.rgb, shot.size, output=outfile)
    print(f"Saved {outfile}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    if command == "mouse":
        live_mouse_position()
    elif command == "shot":
        full_screenshot()
    elif command == "crop":
        left, top, width, height = map(int, sys.argv[2:6])
        outfile = sys.argv[6]
        region_crop(left, top, width, height, outfile)
    else:
        print(__doc__)
