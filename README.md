# Deducto Parkour RL

Train a reinforcement learning agent to speedrun Deducto's parkour course. The
game is closed-source, so the agent plays like a human would: it reads the
screen with computer vision and drives the real keyboard and mouse.

https://user-images.githubusercontent.com/PLACEHOLDER/demo.mp4

<!-- Replace the line above with your video. See "Adding a video" at the bottom. -->

**How it works:** you record demonstrations of yourself playing → the policy is
behavior-cloned from them → PPO fine-tunes it against a reward built from the
on-screen timer. Because the finish is ~12 seconds of precise movement away,
training uses a curriculum: move the in-game finish line to an early platform,
master it, then push it further out.

> Trained runs are a personal technical project. Please don't submit agent times
> to the community leaderboard.

---

## 1. Install

```bash
git clone <your-repo-url> && cd deductoRL
python3 -m venv venv
```

**macOS** (the game runs through CrossOver):
```bash
source venv/bin/activate
pip install -r requirements.txt
brew install tesseract
```

**Windows** (native Steam build):
```powershell
venv\Scripts\activate
pip install -r requirements.txt
```
Then install Tesseract from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
and add it to your PATH.

## 2. Grant permissions

**macOS** — System Settings → Privacy & Security. Grant **Screen Recording**,
**Accessibility**, and **Microphone** to whatever runs Python (Terminal.app, or
your editor if you run it from there). Without these, capture returns black
frames and input silently does nothing.

**Windows** — run your terminal **as Administrator**, otherwise Windows blocks
simulated input to the game.

## 3. Set up the game

- Windowed mode at a fixed resolution. Don't move or resize the window later.
- Bind a **respawn key** that teleports you to the course start, and set
  `RESPAWN_KEY` in `config.py` to match (default `"F"`). Every episode uses it,
  so runs always start from an identical position.
- Move the in-game **finish line to platform 2** for your first training stage.

## 4. Calibrate

```bash
python calibrate.py mouse     # hover the window corners and the timer, note the pixels
```
Put those numbers into `CAPTURE_REGION` and `TIMER_ROI` in `config.py`. Also set
`APP_NAME` (macOS: the process name in Activity Monitor; Windows: the window
title).

```bash
python diagnose_timer.py      # play a full run; it prints recommended values
```
Copy its recommended `TIMER_*` values into `config.py`. The timer must read
reliably — everything downstream depends on it.

### Optional but recommended: audio finish detection

The finish plays a distinct tone, which is far more reliable than watching the
timer freeze. Neither OS can capture another app's audio directly, so route it
through a loopback device:

- **macOS:** `brew install blackhole-2ch`, then in *Audio MIDI Setup* create a
  Multi-Output Device containing both your speakers and BlackHole, and select it
  as your system output.
- **Windows:** install [VB-Cable](https://vb-audio.com/Cable/), set it as your
  output, and enable "Listen to this device" on it so you can still hear the game.

```bash
python diagnose_audio.py --list    # find the loopback input, set AUDIO_DEVICE
python diagnose_audio.py           # play and finish a run
```
Set `AUDIO_FINISH_MIN_ENERGY` and `AUDIO_FINISH_MIN_RATIO` between what normal
play shows and what the finish spikes to, then set `AUDIO_FINISH_ENABLED = True`.

## 5. Record demos

```bash
python record_demos.py --check-input                              # verify input is seen
python record_demos.py --out demos/run --episodes 40 --keep-falls
```

Hold **W** the whole time (the agent always holds forward). Press your respawn
key during each countdown. Aim for 40–80 runs with variety: clean runs on
slightly different lines, runs where you wobble and recover, and some falls.
Recovery runs matter most — they teach the agent what to do when it drifts off
the ideal path.

## 6. Pretrain

```bash
python pretrain_bc.py --demos demos/run --out checkpoints/bc_pretrained.zip
```

No game needed. Watch the validation accuracy — jump ~0.9 and look ~0.6 is healthy.

## 7. Train

Each stage stops itself once the agent finishes 8 of the last 10 episodes.

```bash
# finish line on platform 2
python train.py --name p02 --resume-from checkpoints/bc_pretrained.zip --max-episode-seconds 6
```

Then move the finish line one platform further and resume from the stage you
just finished, keeping the anchor pointed at the behavior-cloned model:

```bash
python train.py --name p03 --resume-from checkpoints/p02/ppo_deducto_parkour_final.zip \
                --anchor-to checkpoints/bc_pretrained.zip --max-episode-seconds 6
```

Repeat for p04, p05, … raising `--max-episode-seconds` as runs get longer.
Press **Shift+Alt+K** to stop early; the model is always saved.

Watch progress with:
```bash
tensorboard --logdir logs
```
`rollout/ep_rew_mean` maps to finish rate: about −5 means it almost never
finishes, 0 is roughly half, +3 or above is mostly finishing.

The machine is unusable while training — the agent drives your real mouse and
keyboard, and refocuses the game every episode.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Episodes end instantly as "fell" | `TIMER_ROI` is wrong or the timer flickers — rerun `diagnose_timer.py` |
| Every episode is "never_started" | The agent can't reach platform 1 in time, or the timer never arms |
| Agent flails wildly | Entropy too high, or PPO erased the clone — lower `ENT_COEF`, raise `BC_ANCHOR_COEF` |
| Agent repeats one identical mistake | Raise `ENT_COEF` toward 0.01, or record demos of that exact section |
| A stage plateaus for tens of thousands of steps | Record ~10 demos of the blocking jump and re-run `pretrain_bc.py` |
| Audio detector reads all zeros | Wrong input device, missing mic permission, or `AUDIO_SAMPLE_RATE` doesn't match the device |

Changing `FRAME_SIZE` or `STEP_DT` invalidates existing demos — re-record them.

## Tuning

Everything lives in `config.py`. The values that matter most:

- `ENT_COEF` — exploration. Raise to discover new terrain, lower to refine.
- `BC_ANCHOR_COEF` — how tightly the agent is held to your demos.
- `FRAME_SIZE` — visual detail. Bigger sees more but needs more demos.
- `SUCCESS_STOP_THRESHOLD` — when a stage is considered mastered.

## Platform support

Developed and tested on macOS with CrossOver. The Windows input backend
(`input_controller.py`, `hotkey.py`, and the `pynput` recorder path) is
implemented but **untested** — issues and PRs welcome.

## Adding a video

1. Record your agent training (QuickTime on macOS, Game Bar on Windows). Keep
   it under 10 MB and use `.mp4`.
2. Open a new issue in your own repo, drag the video into the comment box, and
   wait for it to upload. GitHub replies with a URL like
   `https://user-images.githubusercontent.com/…/demo.mp4`.
3. Paste that URL on its own line near the top of this README (replacing the
   placeholder). GitHub renders bare video URLs as an inline player — don't wrap
   it in markdown image or link syntax.
4. You can close the issue; the uploaded file stays available.

A GIF works too (`![demo](demo.gif)`) and autoplays, but files get large fast.
