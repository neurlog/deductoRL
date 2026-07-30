"""Screen capture. The observation is 2 channels: grayscale plus a "redness"
mask, because the red platforms are the task-relevant objects and grayscale
nearly erases them."""

import cv2
import numpy as np
import mss

import config


class ScreenCapture:
    def __init__(self, region: dict = None):
        self.region = region or config.CAPTURE_REGION
        for name, box in (("CAPTURE_REGION", self.region), ("TIMER_ROI", config.TIMER_ROI)):
            if not box["width"] or not box["height"]:
                raise SystemExit(
                    f"config.{name} is still zero. Run `python calibrate.py mouse` and "
                    f"fill in the pixel box, otherwise capture returns nothing."
                )
        self._sct = mss.mss()

    def grab_raw(self) -> np.ndarray:
        shot = self._sct.grab(self.region)
        return np.array(shot)  # BGRA

    def grab_bgr(self) -> np.ndarray:
        return cv2.cvtColor(self.grab_raw(), cv2.COLOR_BGRA2BGR)

    def to_observation(self, frame_bgr: np.ndarray) -> np.ndarray:
        """CNN-ready (H, W, 2) uint8: grayscale + redness mask."""
        small = cv2.resize(frame_bgr, config.FRAME_SIZE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        b, g, r = cv2.split(small.astype(np.int16))
        redness = np.clip(r - np.maximum(g, b), 0, 255).astype(np.uint8)

        return np.dstack([gray, redness])  # (H, W, 2)

    def crop_timer(self, frame_bgr: np.ndarray) -> np.ndarray:
        roi = config.TIMER_ROI
        pad = getattr(config, "TIMER_ROI_PADDING", 0)
        h, w = frame_bgr.shape[:2]
        # Expand the ROI by `pad` on every side, clamped to the frame so the
        # crop never runs off the captured region.
        y0 = max(0, roi["top"] - pad)
        x0 = max(0, roi["left"] - pad)
        y1 = min(h, roi["top"] + roi["height"] + pad)
        x1 = min(w, roi["left"] + roi["width"] + pad)
        return frame_bgr[y0:y1, x0:x1]

    def grab_observation(self) -> np.ndarray:
        return self.to_observation(self.grab_bgr())

    def grab_timer_crop(self) -> np.ndarray:
        return self.crop_timer(self.grab_bgr())

    def close(self):
        self._sct.close()
