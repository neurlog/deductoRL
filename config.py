"""Settings. Values marked CALIBRATE depend on your screen and game setup."""

# Capture geometry. CALIBRATE with calibrate.py.
CAPTURE_REGION = {"left": 0, "top": 0, "width": 1024, "height": 640}
TIMER_ROI = {"left": 475, "top": 48, "width": 69, "height": 24}
TIMER_ROI_PADDING = 6

# Observation: grayscale + a red-platform mask, stacked 4 deep in train.py.
# Changing FRAME_SIZE invalidates recorded demos.
FRAME_SIZE = (128, 128)
OBS_CHANNELS = 2

# Timer detection. CALIBRATE all four with diagnose_timer.py.
TIMER_THRESHOLD = 150
TIMER_PRESENT_MIN_PIXELS = 25
TIMER_CHANGED_MIN_PIXELS = 20
FALL_DETECT_MISSING_FRAMES = 2

FINISH_FREEZE_SECONDS = 1.0
FINISH_DISAPPEAR_AFTER_FREEZE_SECONDS = 0.5
ARM_TIMEOUT_SECONDS = 20.0
RESTART_IF_NOT_ARMED_SECONDS = 2.0

# The timer only starts once you land on the first platform, so the tracker
# waits for it to tick this many times before treating the run as live.
# Requiring 2 stops a frozen leftover time counting as a new run.
ARM_MIN_CHANGES = 2

KEYS = ("W", "A", "S", "D", "F", "SPACE")
RESPAWN_KEY = "F"           # in-game key that returns you to the course start

STEP_DT = 0.12              # seconds per decision; changing it invalidates demos
JUMP_TAP_DURATION = 0.05

LOOK_DELTAS_X = [-250, -120, -60, -25, -10, 0, 10, 25, 60, 120, 250]
LOOK_DELTAS_Y = [-10, 0, 10]
# Big turns are split into small events so the motion looks like a real mouse
# rather than an instant snap.
MOUSE_MAX_DELTA_PER_EVENT = 40
MOUSE_EVENT_INTERVAL_SECONDS = 0.006

APP_NAME = "Deduction.exe"  # CALIBRATE: process name (macOS) or window title (Windows)

RESPAWN_SETTLE_SECONDS = 0.4
FOCUS_SETTLE_SECONDS = 0.15
POST_RESET_DELAY_SECONDS = 0.5

# Audio finish detection. Needs game audio routed to an input device via a
# loopback driver (see README), then CALIBRATE with diagnose_audio.py.
AUDIO_FINISH_ENABLED = False
AUDIO_DEVICE = None
AUDIO_SAMPLE_RATE = 48000   # must match the device's native rate or you get silence
AUDIO_BLOCK_SIZE = 2048
AUDIO_FINISH_FREQ_LOW = 510
AUDIO_FINISH_FREQ_HIGH = 620
AUDIO_FINISH_MIN_ENERGY = 3.0
AUDIO_FINISH_MIN_RATIO = 15.0

MAX_EPISODE_SECONDS = 60
TIME_PENALTY_PER_STEP = 0.01

FINISH_REWARD = 5.0
# Must exceed (TIMER_RUNNING_REWARD_PER_STEP - TIME_PENALTY_PER_STEP) / STEP_DT,
# or a slower finish scores higher than a fast one.
FINISH_TIME_BONUS_SCALE = 0.15
FINISH_MIN_REWARD = 1.0
FALL_PENALTY = -6.0

# Without a per-step reward while the timer runs, an agent that is going to
# fall anyway scores better by falling sooner. Capped below FINISH_REWARD.
TIMER_RUNNING_REWARD_PER_STEP = 0.02
TIMER_RUNNING_REWARD_MAX = 3.0

# Optical-flow progress signal, mainly for the run-up to the first platform,
# which earns nothing else. Suppressed on big turns so spinning can't farm it.
FLOW_REWARD_SCALE = 0.01
FLOW_MAX_PER_STEP = 1.0
FLOW_SUPPRESS_LOOK_ABS = 60

TOTAL_TIMESTEPS = 200_000
N_STEPS = 512
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
N_EPOCHS = 8
CLIP_RANGE = 0.1
GAMMA = 0.995

# Raise toward 0.01 when the agent is stuck repeating one identical failure;
# lower toward 0.001 when polishing a section it already completes.
ENT_COEF = 0.005

# Pull toward the cloned policy, decaying to zero. Without it, early PPO
# updates erase the cloning; too strong and the agent never improves on it.
BC_ANCHOR_COEF = 0.4
BC_ANCHOR_DECAY_STEPS = 60_000

# Stop a stage once this many of the last N episodes finish.
SUCCESS_STOP_WINDOW = 10
SUCCESS_STOP_THRESHOLD = 8

LOG_DIR = "logs"
CHECKPOINT_DIR = "checkpoints"
SAVE_FREQ = 1_000
