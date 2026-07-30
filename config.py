"""Settings. Anything marked CALIBRATE depends on your screen and game setup."""

# ---------------------------------------------------------------- capture ---
# PUT YOUR NUMBERS HERE. Run `python calibrate.py mouse`, hover the game
# window's corners and the timer, and fill both boxes in. Nothing works until
# you do; the code refuses to start while these are zero.
CAPTURE_REGION = {"left": 0, "top": 0, "width": 0, "height": 0}   # the game window
TIMER_ROI = {"left": 0, "top": 0, "width": 0, "height": 0}        # the on-screen timer
TIMER_ROI_PADDING = 6           # px of slack so the timer can't clip out of the crop

FRAME_SIZE = (128, 128)         # what the CNN sees. Re-record demos if changed
OBS_CHANNELS = 2                # grayscale + red-platform mask

# -------------------------------------------------------- timer detection ---
TIMER_THRESHOLD = 150           # CALIBRATE: brightness cutoff for timer digits
TIMER_PRESENT_MIN_PIXELS = 25   # CALIBRATE: lit pixels meaning "timer on screen"
TIMER_CHANGED_MIN_PIXELS = 20   # CALIBRATE: changed pixels meaning "timer ticked"
FALL_DETECT_MISSING_FRAMES = 2  # frames without a timer before calling it a fall

ARM_MIN_CHANGES = 2             # ticks before the run counts as live. 2 ignores the
                                # frozen time left over from the previous run
FINISH_FREEZE_SECONDS = 1.0     # motionless timer this long = finished (unused if audio is on)
FINISH_DISAPPEAR_AFTER_FREEZE_SECONDS = 0.5   # frozen then gone = finish, not a fall
ARM_TIMEOUT_SECONDS = 20.0      # give up waiting for the run to start
RESTART_IF_NOT_ARMED_SECONDS = 2.0            # training-only: respawn and retry sooner

# ------------------------------------------------------------------ input ---
KEYS = ("W", "A", "S", "D", "F", "SPACE")
RESPAWN_KEY = "F"               # in-game key that returns you to the course start
APP_NAME = "Deduction.exe"      # CALIBRATE: process name (macOS) or window title (Windows)

STEP_DT = 0.12                  # seconds per decision. Re-record demos if changed
JUMP_TAP_DURATION = 0.05        # how long SPACE is held for a jump

LOOK_DELTAS_X = [-250, -120, -60, -25, -10, 0, 10, 25, 60, 120, 250]  # horizontal aim choices
LOOK_DELTAS_Y = [-10, 0, 10]                                          # vertical aim choices
MOUSE_MAX_DELTA_PER_EVENT = 40  # split big turns into steps this size, like a real mouse
MOUSE_EVENT_INTERVAL_SECONDS = 0.006          # gap between those steps

RESPAWN_SETTLE_SECONDS = 0.4    # wait for the teleport to finish
FOCUS_SETTLE_SECONDS = 0.15     # wait for the window to take focus before sending keys
POST_RESET_DELAY_SECONDS = 0.5  # let the agent settle before it starts acting

# ------------------------------------------------------------------ audio ---
# Finish detection by sound, far steadier than watching the timer freeze.
# Route game audio to an input device with a loopback driver (see README).
AUDIO_FINISH_ENABLED = False    # set True once the two lines below are sorted
AUDIO_DEVICE = "INPUT DEVICE HERE"   # `python diagnose_audio.py --list` shows the names
AUDIO_SAMPLE_RATE = 48000       # must match the device, or you record silence
AUDIO_BLOCK_SIZE = 2048         # samples per analysis window
AUDIO_FINISH_FREQ_LOW = 510     # bracket the finish tone's pitch
AUDIO_FINISH_FREQ_HIGH = 620
AUDIO_FINISH_MIN_ENERGY = 3.0   # CALIBRATE: diagnose_audio.py, above what normal play shows
AUDIO_FINISH_MIN_RATIO = 15.0   # CALIBRATE: how much the tone must stand out from other sound

# ----------------------------------------------------------------- reward ---
MAX_EPISODE_SECONDS = 60
TIME_PENALTY_PER_STEP = 0.01    # small cost per step, so dawdling is never free

FINISH_REWARD = 5.0             # base payout for finishing
FINISH_TIME_BONUS_SCALE = 0.15  # reward lost per second taken, so faster runs pay more
FINISH_MIN_REWARD = 1.0         # floor, so a slow finish still beats not finishing
FALL_PENALTY = -6.0             # cost of falling off

FLOW_REWARD_SCALE = 0.01        # credit for scenery streaming past, i.e. moving forward.
FLOW_MAX_PER_STEP = 1.0         # Mainly carries the run-up to platform 1, which pays nothing
FLOW_SUPPRESS_LOOK_ABS = 60     # ignore flow on big turns, else spinning farms it

# --------------------------------------------------------------- training ---
TOTAL_TIMESTEPS = 200_000
N_STEPS = 512                   # steps gathered before each learning update
BATCH_SIZE = 64
N_EPOCHS = 8                    # passes over each batch of gathered steps
LEARNING_RATE = 1e-4
CLIP_RANGE = 0.1                # ceiling on how far one update may move the policy
GAMMA = 0.995                   # how far ahead the agent plans

ENT_COEF = 0.005                # randomness. Raise toward 0.01 when it repeats one
                                # identical failure, lower toward 0.001 to polish
BC_ANCHOR_COEF = 0.4            # pull back toward your demos, fading to nothing over
BC_ANCHOR_DECAY_STEPS = 60_000  # the decay. Too weak and early updates erase the
                                # cloning, too strong and it never improves on you

SUCCESS_STOP_WINDOW = 10        # end the stage once this many of the last N episodes
SUCCESS_STOP_THRESHOLD = 8      # finished, i.e. time to move the finish line out

LOG_DIR = "logs"
CHECKPOINT_DIR = "checkpoints"
SAVE_FREQ = 1_000               # save a checkpoint this often, in steps
