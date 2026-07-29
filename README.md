# DeductoRL

A reinforcement learning agent that learns how to play Deducto parkour. The
game is closed-source, so it reads the screen with computer vision 
and utilizes the real keyboard and mouse.

<p align="center">
  <img width="600" height="386" alt="snipper-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/a6c2d002-0eb6-48d0-b458-3d7d53d6941b"/>
</p>

## How it works: 
This reinforcement learning agent works in a supervised learning manner, meaning you record demos of yourself playing, the policy is
behaviour-cloned from the demos and the agent fine-tunes the demos against a reward built system from
the on-screen timer and finish line. The on-screen timer is used to tell the agent when it has fallen, as it becomes invisible when you fall off, and the finish line is used
as a way to inform the robot when it has finished. 

Due to the finish being more than 4 seconds of precise movement away, the agent utilises curriculum learning: 
moving the in-game finish line to an early platform, mastering it, then pushing the finish further out.

Moving the in-game timer can be done using `DeductoHelper.dll` which can be downloaded from [Deducto-Tools](https://github.com/neurlog/Deducto-Tools). Massive shoutout to Dirty for helping me by creating and editing the DeductoHelper tool. DeductoHelper enables you to move the finish line, bind a respawn key, and set a respawn point.

> I built and tested this on macOS (the game running under CrossOver). The
> Windows code paths are written but have not been run against the real game, so
> expect bugs. For support, you can contact my discord: neurlog.

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
- Respawn key is naturally bound to "F" from `DeductoHelper.dll`. Every episode uses it,
  so runs always begin from an identical position.
- Move the in-game **finish line to platform 2** for your first training stage. Also available on `DeductoHelper.dll`

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

### Audio finish detection

The finish plays a distinct tone, I used this feature to tell my agent when I finish a run. No OS lets you capture another app's audio directly, so route it through
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

Aim for 30-50 runs with variety: clean runs, some sloppy ones and you can use `--keep-falls` if you'd like to save runs where you fall. I kept my falls on so the agent learns what inputs caused me to fail, and so the agent familiarises themself with failed attempts.

## 6. Pretrain

Pretraining teaches the agent to copy you before it ever tries the course
itself. It goes through your demos frame by frame and learns to predict
what you pressed given what was on screen, which is ordinary supervised
learning rather than trial and error.

Copying you first means it starts out roughly knowing the route, and training is then
spent improving on that instead of searching blindly.

```powershell
python pretrain.py --demos demos/run --out checkpoints/bc_pretrained.zip
```

## 7. Train

Each stage stops itself once the agent finishes 8 of the last 10 episodes.

```powershell
python train.py --name p02 --resume-from checkpoints/bc_pretrained.zip --max-episode-seconds 6
```

Then move the finish line one platform further and resume from the stage you
just finished, keeping the anchor pointed at the behavior-cloned model:

```powershell
python train.py --name p03 --resume-from checkpoints/p02/ppo_deducto_parkour_final.zip --anchor-to checkpoints/bc_pretrained.zip --max-episode-seconds 6
```

Repeat for p04, p05 and so on, raising `--max-episode-seconds` as runs get
longer. Press **Shift+Alt+K** to stop early. Stopping early saves the model.

---

## Tuning

Everything lives in `config.py`. The settings worth touching first:

- `ENT_COEF` sets how much the agent explores. Raise it to discover new terrain,
  lower it to refine what already works.
- `BC_ANCHOR_COEF` sets how tightly it is held to your demos.

## Important notes

- Due to the game being closed-source, whilst training the agent, you cannot use your computer :(

