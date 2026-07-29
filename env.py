"""Gymnasium environment wrapping the parkour course.

W is held for the whole episode; the agent chooses jump timing and camera aim.
Action space: [jump(2), look_h(11), look_v(3)].
"""

import time

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config
from capture import ScreenCapture
from input_controller import (
    activate_app,
    begin_action,
    end_jump,
    key_down,
    key_up,
    release_all_movement_keys,
)
from reward import TimerTracker, ocr_timer_seconds


class DeductoParkourEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.capture = ScreenCapture()
        self.action_space = spaces.MultiDiscrete(
            [2, len(config.LOOK_DELTAS_X), len(config.LOOK_DELTAS_Y)]
        )
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(*config.FRAME_SIZE, config.OBS_CHANNELS), dtype=np.uint8,
        )
        self._episode_start_time = None
        self._tracker = None
        self._prev_gray_small = None

        self._audio = None
        if config.AUDIO_FINISH_ENABLED:
            from audio import AudioFinishListener
            listener = AudioFinishListener()
            self._audio = listener if listener.start() else None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        release_all_movement_keys()
        activate_app()

        # Wait for focus BEFORE sending the respawn key, or it silently no-ops
        # and the episode starts from wherever the last one ended.
        time.sleep(config.FOCUS_SETTLE_SECONDS)
        if config.RESPAWN_KEY is not None:
            key_down(config.RESPAWN_KEY)
            time.sleep(0.05)
            key_up(config.RESPAWN_KEY)
        time.sleep(config.RESPAWN_SETTLE_SECONDS)
        time.sleep(config.POST_RESET_DELAY_SECONDS)

        self._episode_start_time = time.time()
        self._prev_gray_small = None
        # With audio on, the timer is a pure arm+fall detector; the tone owns
        # finishes, so its unreliable freeze heuristic is switched off.
        self._tracker = TimerTracker(
            start_time=self._episode_start_time,
            detect_finish=(self._audio is None),
        )
        if self._audio is not None:
            self._audio.reset()

        frame = self.capture.grab_bgr()
        self._tracker.update(self.capture.crop_timer(frame))
        key_down("W")
        return self.capture.to_observation(frame), {}

    def _flow_reward(self, gray_small, look_dx):
        if config.FLOW_REWARD_SCALE <= 0:
            return 0.0
        prev = self._prev_gray_small
        self._prev_gray_small = gray_small
        if prev is None or abs(look_dx) >= config.FLOW_SUPPRESS_LOOK_ABS:
            return 0.0
        h = gray_small.shape[0] // 2
        flow = cv2.calcOpticalFlowFarneback(prev[h:], gray_small[h:], None,
                                            0.5, 2, 9, 2, 5, 1.1, 0)
        mag = float(np.mean(np.linalg.norm(flow, axis=2)))
        return config.FLOW_REWARD_SCALE * min(mag, config.FLOW_MAX_PER_STEP)

    def step(self, action):
        step_deadline = time.time() + config.STEP_DT
        jump_idx, look_h_idx, look_v_idx = [int(a) for a in action]
        action_start = time.time()
        look_dx = begin_action(jump_idx, look_h_idx, look_v_idx)

        if jump_idx == 1:
            remaining = action_start + config.JUMP_TAP_DURATION - time.time()
            if remaining > 0:
                time.sleep(remaining)
            end_jump()

        frame = self.capture.grab_bgr()
        obs = self.capture.to_observation(frame)
        timer_crop = self.capture.crop_timer(frame)

        reward = -config.TIME_PENALTY_PER_STEP
        terminated = truncated = False
        info = {}
        elapsed = time.time() - self._episode_start_time

        event = self._tracker.update(timer_crop)

        if self._audio is not None and self._tracker.armed and self._audio.finished():
            event = "finish"

        # If the run never started, end now so reset() respawns and retries.
        if (event is None and not self._tracker.armed
                and elapsed > config.RESTART_IF_NOT_ARMED_SECONDS):
            event = "never_started"

        if event in ("fell", "never_started"):
            reward += config.FALL_PENALTY
            terminated = True
            info["event"] = event
        elif event == "finish":
            final_time = ocr_timer_seconds(timer_crop)
            if final_time is None or final_time <= 0:
                final_time = elapsed
            reward += max(
                config.FINISH_REWARD - final_time * config.FINISH_TIME_BONUS_SCALE,
                config.FINISH_MIN_REWARD,
            )
            terminated = True
            info["event"] = "finish"
            info["elapsed_seconds"] = final_time
        else:
            reward += self._flow_reward(obs[..., 0], look_dx)

        info["armed"] = self._tracker.armed
        if not terminated and elapsed > config.MAX_EPISODE_SECONDS:
            truncated = True

        # Every step lasts exactly STEP_DT so identical actions have identical
        # physical consequences.
        remaining = step_deadline - time.time()
        if remaining > 0:
            time.sleep(remaining)

        return obs, reward, terminated, truncated, info

    def close(self):
        release_all_movement_keys()
        self.capture.close()
        if self._audio is not None:
            self._audio.stop()
