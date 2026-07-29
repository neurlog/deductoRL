# Deducto Parkour RL

Train a reinforcement learning agent to speedrun Deducto's parkour course. The
game is closed-source, so the agent plays like a human would: it reads the
screen with computer vision and drives the real keyboard and mouse.

https://user-images.githubusercontent.com/PLACEHOLDER/demo.mp4

<!-- Replace the line above with your video. See "Adding a video" at the bottom. -->

**How it works:** you record demonstrations of yourself playing, the policy is
behavior-cloned from them, then PPO fine-tunes it against a reward built from
the on-screen timer. Because the finish is ~12 seconds of precise movement away,
training uses a curriculum: move the in-game finish line to an early platform,
master it, then push it further out.

> **Status:** built and tested on macOS (the game running under CrossOver). The
> Windows code paths are written but have not been run against the real game, so
> expect to hit rough edges, most likely in demo recording. Bug reports and PRs
> are very welcome.

> Trained runs are a personal technical project. Please don't submit agent times
> to the community leaderboard.

---

## 1. Install

```powershell
git clone <your-repo-url>
cd deductoRL
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Install Tesseract from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
and tick "Add to PATH" during setup. Verify with `tesseract --version`.

<details>
<summary>macOS (game running under CrossOver)</summary>

```bash
git clone <your-repo-url> && cd deductoRL
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew install tesseract
```
</details>

## 2. Permissions

Run your terminal **as Administrator**. Windows blocks simulated input to games
from an unprivileged process, so without this the agent presses keys and nothing
happens.

<details>
<summary>macOS</summary>

System Settings -> Privacy & Security. Grant **Screen Recording**,
**Accessibility**, and **Microphone** to whatever runs Python (Terminal.app, or
your editor if you launch it from there). Without these, capture returns black
frames and input silently does nothing.
</details>

## 3. Set up the game

- Windowed mode at a fixed resolution. Don't move or resize the window afterwards.
- Bind a **respawn key** that returns you to the course start, and set
  `RESPAWN_KEY` in `config.py` to match (default `"F"`). Every episode uses it,
  so runs always begin from an identical position.
- Move the in-game **finish line to platform 2** for your first training stage.

## 4. Calibrate

```powershell
python calibrate.py mouse
```

Hover the game window's corners and the timer, note the pixel coordinates, and
put them into `CAPTURE_REGION` and `TIMER_ROI` in `config.py`. Set `APP_NAME` to
the game's window title (on macOS, the process name from Activity Monitor).

```powershell
python diagnose_timer.py
```

Play a full run while this watches, then copy its recommended `TIMER_*` values
into `config.py`. The timer has to read reliably; everything downstream depends
on it.

### Audio finish detection

The finish plays a distinct tone, which is far steadier than watching the timer
freeze. No OS lets you capture another app's audio directly, so route it through
a loopback device.

Install [VB-Cable](https://vb-audio.com/Cable/), set it as your default output,
then in Sound settings enable **Listen to this device** on the CABLE input so you
can still hear the game.

<details>
<summary>macOS</summary>

`brew install blackhole-2ch`, then in *Audio MIDI Setup* create a Multi-Output
Device containing both your speakers and BlackHole, and select it as your system
output.
</details>

```powershell
python diagnose_audio.py --list    # find the loopback input, set AUDIO_DEVICE
python diagnose_audio.py           # play and finish a run
```

Set `AUDIO_FINISH_MIN_ENERGY` and `AUDIO_FINISH_MIN_RATIO` between what normal
play shows and what the finish spikes to, then set `AUDIO_FINISH_ENABLED = True`.

## 5. Record demos

```powershell
python record_demos.py --check-input
python record_demos.py --out demos/run --episodes 40 --keep-falls
```

Run `--check-input` first and confirm the numbers move when you move the mouse.

Aim for 40-80 runs with variety: clean runs and some sloppy ones. It means the agent will be more versatile.

## 6. Pretrain

```powershell
python pretrain.py --demos demos/run --out checkpoints/bc_pretrained.zip
```

## 7. Train

Each stage stops itself once the agent finishes 8 of the last 10 episodes.

```powershell
python train.py --name p02 --resume-from checkpoints/bc_pretrained.zip --max-episode-seconds 6
```

Then move the finish line one platform further out and resume from the stage you
just finished, keeping the anchor pointed at the behavior-cloned model:

```powershell
python train.py --name p03 --resume-from checkpoints/p02/ppo_deducto_parkour_final.zip --anchor-to checkpoints/bc_pretrained.zip --max-episode-seconds 6
```

Repeat for p04, p05 and so on, raising `--max-episode-seconds` as runs get
longer. Press **Shift+Alt+K** to stop early; the model is always saved.

---

## Tuning

Everything lives in `config.py`. The settings worth touching first:

- `ENT_COEF` sets how much the agent explores. Raise it to discover new terrain,
  lower it to refine what already works.
- `BC_ANCHOR_COEF` sets how tightly it is held to your demos.
