"""PPO training, one curriculum stage per run.

    python train.py --name p02 --resume-from checkpoints/bc_pretrained.zip
    python train.py --name p03 --resume-from checkpoints/p02/ppo_deducto_parkour_final.zip \
                    --anchor-to checkpoints/bc_pretrained.zip

Each stage stops itself once the success gate is met (see config).
"""

import argparse
import os
import time
from collections import deque

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

import config
from env import DeductoParkourEnv
from hotkey import is_stop_hotkey_pressed
from input_controller import key_down, release_all_movement_keys


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on a Deducto parkour segment.")
    parser.add_argument(
        "--name", default="run",
        help="Name for this segment/run — organizes checkpoints and logs into their own subfolder."
    )
    parser.add_argument(
        "--resume-from", default=None,
        help="Path to a previously saved model .zip to continue training from."
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override config.TOTAL_TIMESTEPS just for this run."
    )
    parser.add_argument(
        "--max-episode-seconds", type=float, default=None,
        help="Override config.MAX_EPISODE_SECONDS just for this run."
    )
    parser.add_argument(
        "--anchor-to", default=None,
        help="Path to a BC checkpoint to KL-anchor the policy to during "
             "fine-tuning. Defaults to --resume-from. Pass --anchor-coef 0 "
             "to disable."
    )
    parser.add_argument(
        "--anchor-coef", type=float, default=config.BC_ANCHOR_COEF,
        help="Initial strength of the KL anchor (decays to 0 over "
             "config.BC_ANCHOR_DECAY_STEPS). 0 disables."
    )
    return parser.parse_args()


class HotkeyStopCallback(BaseCallback):
    """Stop on Shift+Alt+K, saving both the normal resume point and a
    timestamped snapshot you can always roll back to."""

    def __init__(self, checkpoint_dir: str):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir

    def _on_step(self) -> bool:
        if is_stop_hotkey_pressed():
            snapshot = os.path.join(
                self.checkpoint_dir,
                f"ppo_deducto_parkour_stopped_{self.num_timesteps}steps_{int(time.time())}",
            )
            self.model.save(snapshot)
            print(f"\n[train] Shift+Option+K — saved snapshot {snapshot}.zip; stopping "
                  "(final checkpoint written on exit too).")
            return False
        return True


class FinishCounterCallback(BaseCallback):
    """Prints a running tally every time the agent actually finishes a run."""

    def __init__(self):
        super().__init__()
        self.num_finishes = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info.get("event") == "finish":
                self.num_finishes += 1
                print(f"Finish: {self.num_finishes}")
        return True


class FreezeDuringUpdateCallback(BaseCallback):
    """The game keeps running during PPO's multi-second update, so release the
    keys or the character sprints off the course unattended every rollout."""

    def _on_rollout_end(self) -> None:
        release_all_movement_keys()

    def _on_rollout_start(self) -> None:
        key_down("W")

    def _on_step(self) -> bool:
        return True


class SuccessGateCallback(BaseCallback):
    """Stop once `threshold` of the last `window` episodes finished — the
    signal to move the finish line to the next platform."""

    def __init__(self, window: int, threshold: int):
        super().__init__()
        self.window = window
        self.threshold = threshold
        self.outcomes = deque(maxlen=window)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:   # Monitor adds this key at episode end
                continue
            self.outcomes.append(1 if info.get("event") == "finish" else 0)
            wins = sum(self.outcomes)
            if len(self.outcomes) == self.window and wins >= self.threshold:
                print(f"\n[train] SUCCESS GATE: {wins}/{self.window} of the last "
                      f"episodes finished — stage mastered. Stopping and saving; "
                      f"move the finish line to the next platform and resume "
                      f"from this run's ppo_deducto_parkour_final.zip.")
                return False
        return True


class BCAnchorCallback(BaseCallback):
    """KL anchor to the frozen BC policy, decaying to zero.

    Early on every episode fails, so advantages are uniformly negative and PPO
    would otherwise dissolve the cloned behavior into noise within a couple of
    updates. This pulls it back after each rollout.
    """

    def __init__(self, anchor_path: str, coef: float, decay_steps: int, batch_size: int = 64):
        super().__init__()
        self.anchor_path = anchor_path
        self.coef = coef
        self.decay_steps = decay_steps
        self.batch_size = batch_size
        self._anchor = None

    def _on_training_start(self) -> None:
        anchor = PPO.load(self.anchor_path, device=self.model.device)
        self._anchor = anchor.policy
        self._anchor.set_training_mode(False)
        for p in self._anchor.parameters():
            p.requires_grad_(False)
        print(f"[train] KL-anchoring to {self.anchor_path} "
              f"(coef {self.coef}, decaying over {self.decay_steps} steps)")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        coef = self.coef * max(0.0, 1.0 - self.model.num_timesteps / self.decay_steps)
        self.logger.record("train/bc_anchor_coef", coef)
        if coef <= 0.0:
            return

        obs = self.model.rollout_buffer.observations  # (n_steps, n_envs, C, H, W)
        obs = obs.reshape(-1, *obs.shape[2:])
        policy = self.model.policy
        policy.set_training_mode(True)

        kls = []
        idx = np.random.permutation(len(obs))
        for start in range(0, len(obs), self.batch_size):
            batch = torch.as_tensor(
                obs[idx[start : start + self.batch_size]], device=self.model.device
            )
            with torch.no_grad():
                anchor_heads = self._anchor.get_distribution(batch).distribution
            current_heads = policy.get_distribution(batch).distribution
            kl = sum(
                torch.distributions.kl_divergence(a, c).mean()
                for a, c in zip(anchor_heads, current_heads)
            )
            policy.optimizer.zero_grad()
            (coef * kl).backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            policy.optimizer.step()
            kls.append(kl.item())

        self.logger.record("train/bc_anchor_kl", float(np.mean(kls)))


def main():
    args = parse_args()

    if args.max_episode_seconds is not None:
        config.MAX_EPISODE_SECONDS = args.max_episode_seconds

    log_dir = os.path.join(config.LOG_DIR, args.name)
    checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, args.name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Monitor MUST wrap the raw env for episode stats to reach TensorBoard.
    def make_env():
        return Monitor(DeductoParkourEnv(), filename=os.path.join(log_dir, "monitor"))

    vec_env = DummyVecEnv([make_env])
    vec_env = VecFrameStack(vec_env, n_stack=4)

    if args.resume_from:
        print(f"Resuming from {args.resume_from}")
        # custom_objects matters: PPO.load restores the hyperparameters
        # stored INSIDE the zip, so without these overrides your current
        # config.py values are silently ignored on every resume — the BC
        # checkpoint carries whatever LR/ent_coef it was built with.
        model = PPO.load(
            args.resume_from,
            env=vec_env,
            tensorboard_log=log_dir,
            custom_objects={
                "learning_rate": config.LEARNING_RATE,
                "ent_coef": config.ENT_COEF,
                "clip_range": config.CLIP_RANGE,
                "n_epochs": config.N_EPOCHS,
            },
        )
    else:
        model = PPO(
            "CnnPolicy",
            vec_env,
            n_steps=config.N_STEPS,
            batch_size=config.BATCH_SIZE,
            learning_rate=config.LEARNING_RATE,
            gamma=config.GAMMA,
            ent_coef=config.ENT_COEF,
            clip_range=config.CLIP_RANGE,
            n_epochs=config.N_EPOCHS,
            verbose=1,
            tensorboard_log=log_dir,
        )

    callback_list = [
        # First in the list so the keys are released before the other
        # rollout-end work (notably the BC anchor pass) also runs.
        FreezeDuringUpdateCallback(),
        CheckpointCallback(
            save_freq=config.SAVE_FREQ,
            save_path=checkpoint_dir,
            name_prefix="ppo_deducto_parkour",
        ),
        HotkeyStopCallback(checkpoint_dir),
        FinishCounterCallback(),
    ]

    if config.SUCCESS_STOP_THRESHOLD > 0:
        callback_list.append(SuccessGateCallback(
            window=config.SUCCESS_STOP_WINDOW,
            threshold=config.SUCCESS_STOP_THRESHOLD,
        ))

    anchor_path = args.anchor_to or args.resume_from
    if anchor_path and args.anchor_coef > 0:
        callback_list.append(BCAnchorCallback(
            anchor_path=anchor_path,
            coef=args.anchor_coef,
            decay_steps=config.BC_ANCHOR_DECAY_STEPS,
        ))

    callbacks = CallbackList(callback_list)

    total_timesteps = args.timesteps or config.TOTAL_TIMESTEPS

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=(args.resume_from is None),
        )
    finally:
        model.save(os.path.join(checkpoint_dir, "ppo_deducto_parkour_final"))
        vec_env.close()


if __name__ == "__main__":
    main()
