"""Calibrate audio finish detection.

    python diagnose_audio.py --list    # find your loopback input device
    python diagnose_audio.py           # play, finish a run, watch the readout

Set MIN_ENERGY/MIN_RATIO between what normal play shows and what the finish
spikes to, so DETECT reads 1 only when you actually cross the line.
"""

import argparse
import time

import config
from audio import AudioFinishListener, is_finish_tone

try:
    import sounddevice as sd
except Exception:
    sd = None


def main():
    parser = argparse.ArgumentParser(description="Calibrate audio finish detection.")
    parser.add_argument("--list", action="store_true", help="List audio devices and exit.")
    args = parser.parse_args()

    if sd is None:
        raise SystemExit("sounddevice not installed. Run: pip install sounddevice")

    if args.list:
        print(sd.query_devices())
        print("\nPick the INPUT device carrying game audio (e.g. 'BlackHole 2ch')")
        print("and set config.AUDIO_DEVICE to its name or index.")
        return

    listener = AudioFinishListener()
    if not listener.start():
        raise SystemExit("Could not open the audio device — see the message above.")

    print(f"Listening on device {config.AUDIO_DEVICE!r}. Play a run and finish it.")
    print(f"band=[{config.AUDIO_FINISH_FREQ_LOW}-{config.AUDIO_FINISH_FREQ_HIGH}]Hz  "
          f"MIN_ENERGY={config.AUDIO_FINISH_MIN_ENERGY}  MIN_RATIO={config.AUDIO_FINISH_MIN_RATIO}")
    print("One line updates in place. 'max' peak-HOLDS the loudest in-band moment so\n"
          "you can read off the finish spike even after it passes. Ctrl+C to stop.\n")
    max_band = max_ratio = peak_at_max = 0.0
    try:
        while True:
            time.sleep(0.1)
            band, ratio, peak = listener.stats()
            if band > max_band:
                max_band, peak_at_max = band, peak
            max_ratio = max(max_ratio, ratio)
            detect = is_finish_tone(band, ratio)
            line = (f"now  band={band:7.2f} ratio={ratio:6.1f} peak={peak:6.0f}Hz DET={int(detect)}"
                    f"    |    max  band={max_band:7.2f} ratio={max_ratio:6.1f} @{peak_at_max:6.0f}Hz")
            print("\r" + line + "   ", end="", flush=True)
    except KeyboardInterrupt:
        listener.stop()
        print("\n\nDone.")
        print(f"Loudest in-band moment: band={max_band:.2f} ratio={max_ratio:.1f} "
              f"at {peak_at_max:.0f} Hz.")
        if max_band < 0.001:
            print("band never rose above ~0 -> NO audio is reaching this device "
                  "(routing/permission), not a threshold problem. See the checks below.")
        else:
            print("If that spike happened at your finish, set AUDIO_FINISH_MIN_ENERGY just "
                  "under that band value and AUDIO_FINISH_MIN_RATIO just under that ratio.")


if __name__ == "__main__":
    main()
