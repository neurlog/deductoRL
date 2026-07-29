"""Audio finish detection: listens for the game's finish tone.

Far more reliable than watching the timer freeze. Requires the game's audio
routed into an INPUT device via a loopback driver, since neither OS lets you
capture another app's output directly — see the README for setup, then
calibrate with diagnose_audio.py.
"""

import threading
import time

import numpy as np

import config

try:
    import sounddevice as sd
except Exception:  # not installed, or PortAudio missing
    sd = None


def analyze(block: np.ndarray, sample_rate: int):
    """Return (band_energy, ratio, peak_hz) for one audio block.

    band_energy: mean FFT magnitude inside [FREQ_LOW, FREQ_HIGH] — the raw
                 loudness of the high band (calibrate MIN_ENERGY against this).
    ratio:       in-band mean / out-of-band mean. A high tone concentrates
                 energy up high and leaves the rest near silent -> large ratio;
                 broadband noise is flat -> ratio ~1; a low tone -> ratio ~0.
                 So this cleanly separates "a high tone" from "loud noise".
    peak_hz:     frequency of the single loudest bin (handy for calibration).
    """
    x = np.asarray(block, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0, 0.0, 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    freqs = np.fft.rfftfreq(x.size, 1.0 / sample_rate)
    band = (freqs >= config.AUDIO_FINISH_FREQ_LOW) & (freqs <= config.AUDIO_FINISH_FREQ_HIGH)
    if not band.any() or not (~band).any():
        return 0.0, 0.0, 0.0
    in_energy = float(spec[band].mean())
    out_energy = float(spec[~band].mean()) + 1e-9
    peak_hz = float(freqs[int(np.argmax(spec))])
    return in_energy, in_energy / out_energy, peak_hz


def is_finish_tone(band_energy: float, ratio: float) -> bool:
    return (band_energy >= config.AUDIO_FINISH_MIN_ENERGY
            and ratio >= config.AUDIO_FINISH_MIN_RATIO)


class AudioFinishListener:
    """Background listener that timestamps the last finish-tone detection.

    Usage:
        a = AudioFinishListener()
        a.start()                 # opens the input stream (no-op if unavailable)
        ...
        a.reset()                 # at each episode start, clears stale detections
        if a.finished(): ...      # true once the tone fired since reset()
        a.stop()
    """

    def __init__(self):
        self._stream = None
        self._lock = threading.Lock()
        self._finish_at = 0.0
        self._reset_at = time.time()
        self._last = (0.0, 0.0, 0.0)  # band_energy, ratio, peak_hz (diagnostics)
        self.ok = False

    def start(self) -> bool:
        if sd is None:
            print("[audio] sounddevice not installed (`pip install sounddevice`). "
                  "Audio finish detection is OFF.")
            return False
        try:
            self._stream = sd.InputStream(
                device=config.AUDIO_DEVICE,
                channels=1,
                samplerate=config.AUDIO_SAMPLE_RATE,
                blocksize=config.AUDIO_BLOCK_SIZE,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
        except Exception as e:
            print(f"[audio] could not open input device {config.AUDIO_DEVICE!r}: {e}\n"
                  "        Audio finish detection is OFF. Run "
                  "`python diagnose_audio.py --list` to find the right device.")
            self._stream = None
            return False
        self.ok = True
        return True

    def _callback(self, indata, frames, time_info, status):
        band_energy, ratio, peak_hz = analyze(indata[:, 0], config.AUDIO_SAMPLE_RATE)
        with self._lock:
            self._last = (band_energy, ratio, peak_hz)
            if is_finish_tone(band_energy, ratio):
                self._finish_at = time.time()

    def reset(self):
        with self._lock:
            self._reset_at = time.time()

    def finished(self) -> bool:
        with self._lock:
            return self._finish_at > self._reset_at

    def stats(self):
        """(band_energy, ratio, peak_hz) from the most recent block."""
        with self._lock:
            return self._last

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self.ok = False
