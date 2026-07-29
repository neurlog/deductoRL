"""Central configuration. Values marked CALIBRATE are specific to your setup."""

# --- Capture geometry -------------------------------------------------------
# CALIBRATE with `python calibrate.py mouse`. Pixel box of the game window.
CAPTURE_REGION = {"left": 0, "top": 0, "width": 1024, "height": 640}
# CALIBRATE. Pixel box of the on-screen run timer, relative to CAPTURE_REGION.
TIMER_ROI = {"left": 475, "top": 48, "width": 69, "height": 24}
TIMER_ROI_PADDING = 6

# Observation fed to the CNN: grayscale + a "redness" mask that highlights the
# red platforms. Changing FRAME_SIZE invalidates existing demos — re-record.
FRAME_SIZE = (128, 128)
OBS_CHANNELS = 2

# --- Timer detection --------------------------------------------------------
# CALIBRATE all four with `python diagnose_timer.py`, which measures your
# capture noise floor and prints recommended values.
TIMER_THRESHOLD = 150
TIMER_PRESENT_MIN_PIXELS = 25
TIMER_CHANGED_MIN_PIXELS = 20
FALL_DETECT_MISSING_FRAMES = 2

FINISH_FREEZE_SECONDS = 1.0
FINISH_DISAPPEAR_AFTER_FREEZE_SECONDS = 0.5
ARM_TIMEOUT_SECONDS = 20.0
RESTART_IF_NOT_ARMED_SECONDS = 2.0

# The run does not start when the episode does: you spawn, run to the first
# platform, and only then does the timer start. The tracker therefore waits to
# "arm" until it sees the timer TICK this many times. Requiring 2 stops a
# frozen leftover time from the previous run counting as a new run.
ARM_MIN_CHANGES = 2

# --- Input ------------------------------------------------------------------
# Key names are resolved per-platform in input_controller.py.
KEYS = ("W", "A", "S", "D", "F", "SPACE")
# In-game key that respawns you at the course start, so every episode begins
# from an identical position. Set to None if your game has no such key.
RESPAWN_KEY = "F"

STEP_DT = 0.12                  # seconds per decision; changing it invalidates demos
JUMP_TAP_DURATION = 0.05

# Camera deltas per step. Large turns are split into several small events so
# the motion resembles a real mouse rather than an instant snap.
LOOK_DELTAS_X = [-250, -120, -60, -25, -10, 0, 10, 25, 60, 120, 250]
LOOK_DELTAS_Y = [-10, 0, 10]
MOUSE_MAX_DELTA_PER_EVENT = 40
MOUSE_EVENT_INTERVAL_SECONDS = 0.006

# CALIBRATE: the game's process name, used to focus the window.
# macOS/CrossOver: the name shown in Activity Monitor. Windows: the window title.
APP_NAME = "Deduction.exe"

RESPAWN_SETTLE_SECONDS = 0.4
FOCUS_SETTLE_SECONDS = 0.15
POST_RESET_DELAY_SECONDS = 0.5

# --- Audio finish detection -------------------------------------------------
# Far more reliable than watching the timer freeze. Needs the game's audio
# routed into an INPUT device via a loopback driver (see README), then
# CALIBRATE with `python diagnose_audio.py`.
AUDIO_FINISH_ENABLED = False
AUDIO_DEVICE = None             # input device name, e.g. "BlackHole 16ch"
AUDIO_SAMPLE_RATE = 48000       # MUST match the device's native rate or you get silence
AUDIO_BLOCK_SIZE = 2048
AUDIO_FINISH_FREQ_LOW = 510     # Deducto's finish tone peaks near 562 Hz
AUDIO_FINISH_FREQ_HIGH = 620
AUDIO_FINISH_MIN_ENERGY = 3.0
AUDIO_FINISH_MIN_RATIO = 15.0

# --- Episode and reward -----------------------------------------------------
MAX_EPISODE_SECONDS = 60
TIME_PENALTY_PER_STEP = 0.01

FINISH_REWARD = 5.0
# Must exceed (TIMER_RUNNING_REWARD_PER_STEP - TIME_PENALTY_PER_STEP) / STEP_DT,
# or dawdling to a slower finish out-scores finishing fast.
FINISH_TIME_BONUS_SCALE = 0.15
FINISH_MIN_REWARD = 1.0

FALL_PENALTY = -6.0

# Small per-step reward while the timer runs. Without it, an agent that is
# going to fall anyway scores higher by falling sooner. Capped below
# FINISH_REWARD so finishing always beats loitering.
TIMER_RUNNING_REWARD_PER_STEP = 0.02
TIMER_RUNNING_REWARD_MAX = 3.0

# Dense progress signal from optical flow, mainly for the pre-timer approach to
# the first platform, which earns no other reward. Suppressed on big turns so
# spinning in place can't farm it.
FLOW_REWARD_SCALE = 0.01
FLOW_MAX_PER_STEP = 1.0
FLOW_SUPPRESS_LOOK_ABS = 60

# --- Training ---------------------------------------------------------------
TOTAL_TIMESTEPS = 200_000
N_STEPS = 512
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
N_EPOCHS = 8
CLIP_RANGE = 0.1
GAMMA = 0.995

# Raise toward 0.01 when the agent must DISCOVER new terrain and gets stuck
# repeating one identical failure; lower toward 0.001 when polishing a section
# it already completes.
ENT_COEF = 0.005

# Pulls the policy toward the frozen behavior-cloned model, decaying to zero.
# Without it, early PPO updates (where every episode fails) erase the cloned
# behavior. Too strong and it pins the agent at demo skill forever.
BC_ANCHOR_COEF = 0.4
BC_ANCHOR_DECAY_STEPS = 60_000

# Stop a curriculum stage automatically once this many of the last N episodes
# finish — the signal to move the finish line to the next platform.
SUCCESS_STOP_WINDOW = 10
SUCCESS_STOP_THRESHOLD = 8

LOG_DIR = "logs"
CHECKPOINT_DIR = "checkpoints"
SAVE_FREQ = 1_000
