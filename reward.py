"""Timer-HUD tracking: the termination signal when we have no game state.

The run does not begin when the episode does, so the tracker ARMS only once the
timer is visibly ticking, then detects falls (timer vanishes) and, unless audio
detection is handling it, finishes (timer stops changing).
"""

import re
import time

import cv2
import numpy as np
import pytesseract

import config


# --------------------------------------------------------------------------
# Low-level primitives
# --------------------------------------------------------------------------
def threshold_timer(timer_crop_bgr) -> np.ndarray:
    """Binarize the timer ROI: bright digits -> 255, background -> 0."""
    gray = cv2.cvtColor(timer_crop_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, config.TIMER_THRESHOLD, 255, cv2.THRESH_BINARY)
    return thresh


def white_pixels(thresh: np.ndarray) -> int:
    return int(cv2.countNonZero(thresh))


def diff_pixels(prev_thresh, thresh) -> int:
    """How many pixels changed since the previous thresholded ROI."""
    if prev_thresh is None or prev_thresh.shape != thresh.shape:
        return -1  # unknown / no baseline yet
    return int(cv2.countNonZero(cv2.absdiff(prev_thresh, thresh)))


def timer_present(thresh: np.ndarray) -> bool:
    return white_pixels(thresh) >= config.TIMER_PRESENT_MIN_PIXELS


def timer_changed(prev_thresh, thresh) -> bool:
    d = diff_pixels(prev_thresh, thresh)
    if d < 0:
        return True  # no baseline yet
    return d >= config.TIMER_CHANGED_MIN_PIXELS


def ocr_timer_seconds(timer_crop_bgr):
    """Full OCR read. Slow (~120ms), so call sparingly (once, at finish)."""
    gray = cv2.cvtColor(timer_crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, config.TIMER_THRESHOLD, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(
        thresh, config="--psm 7 -c tessedit_char_whitelist=0123456789:."
    ).strip()

    match = re.match(r"(\d+):(\d+(?:\.\d+)?)", text)  # mm:ss.xx
    if match:
        minutes, seconds = match.groups()
        return int(minutes) * 60 + float(seconds)

    match = re.match(r"(\d+(?:\.\d+)?)", text)  # plain seconds
    if match:
        return float(match.group(1))

    return None


# --------------------------------------------------------------------------
# State machine
# --------------------------------------------------------------------------
class TimerTracker:
    """Tracks timer-HUD state across a single episode.

    Shared by env.py and record_demos.py so training and demo recording can
    never drift apart in how they decide an episode ended.

    Usage:
        tracker = TimerTracker()
        event = tracker.update(timer_crop_bgr)  # None | "fell" | "finish"
                                                #      | "never_started"
    """

    def __init__(self, start_time=None, detect_finish=True):
        self.start_time = start_time if start_time is not None else time.time()
        self.detect_finish = detect_finish
        self.armed = False              # has the run actually begun?
        self.armed_at = None
        self.prev_thresh = None
        self.missing_streak = 0
        self.last_change_time = None    # last time the timer visibly changed
        self.frozen_since = None        # when the current freeze began
        self.event = None
        self.change_count = 0           # timer changes seen while pre-run
        self.last_white = 0
        self.last_diff = -1

    @property
    def freeze_seconds(self) -> float:
        if self.frozen_since is None:
            return 0.0
        return time.time() - self.frozen_since

    def update(self, timer_crop_bgr):
        """Feed one frame's timer crop. Returns a terminal event or None."""
        now = time.time()
        thresh = threshold_timer(timer_crop_bgr)
        present = timer_present(thresh)
        changed = timer_changed(self.prev_thresh, thresh)

        self.last_white = white_pixels(thresh)
        self.last_diff = diff_pixels(self.prev_thresh, thresh)
        has_baseline = self.prev_thresh is not None
        self.prev_thresh = thresh

        # ---------------- pre-run: wait for the timer to start ------------
        if not self.armed:
            # Require the timer to actually TICK, not just appear: a frozen
            # leftover time from the last run produces a single change.
            if present and changed and has_baseline:
                self.change_count += 1
                if self.change_count >= config.ARM_MIN_CHANGES:
                    self.armed = True
                    self.armed_at = now
                    self.last_change_time = now
                    self.frozen_since = None
                return None

            if now - self.start_time > config.ARM_TIMEOUT_SECONDS:
                self.event = "never_started"
                return self.event
            return None

        # ---------------- live run ----------------------------------------
        if present:
            self.missing_streak = 0
            if changed:
                self.last_change_time = now
                self.frozen_since = None
            elif self.detect_finish:
                if self.frozen_since is None:
                    self.frozen_since = now
                if now - self.frozen_since >= config.FINISH_FREEZE_SECONDS:
                    self.event = "finish"
                    return self.event
        else:
            self.missing_streak += 1
            # "Froze then vanished" = a finish that hid the HUD, not a fall.
            was_frozen = (
                self.detect_finish
                and self.frozen_since is not None
                and (now - self.frozen_since) >= config.FINISH_DISAPPEAR_AFTER_FREEZE_SECONDS
            )
            if self.missing_streak >= config.FALL_DETECT_MISSING_FRAMES:
                self.event = "finish" if was_frozen else "fell"
                return self.event

        return None
