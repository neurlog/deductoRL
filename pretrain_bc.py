"""Behavior cloning: train the policy to imitate recorded demos, then save a
normal PPO .zip that train.py can --resume-from.

Also pretrains the value head on the demos' actual returns — without it PPO
starts with a random critic whose advantage estimates wreck the cloned policy.

    python pretrain_bc.py --demos demos/run --out checkpoints/bc_pretrained.zip
"""

import argparse
import glob
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

import config

N_STACK = 4


class SpecEnv(gym.Env):
    """Declares the observation/action spaces only.

    Lets us build a PPO model with the exact architecture train.py will
    later expect, without needing the game running to pretrain.
    """

    def __init__(self):
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(*config.FRAME_SIZE, config.OBS_CHANNELS), dtype=np.uint8
        )
        self.action_space = spaces.MultiDiscrete(
            [2, len(config.LOOK_DELTAS_X), len(config.LOOK_DELTAS_Y)]
        )

    def reset(self, *, seed=None, options=None):
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.uint8), 0.0, False, False, {}


def stack_episode(obs: np.ndarray) -> np.ndarray:
    """(T, H, W, C) single frames -> (T, N_STACK*C, H, W) channels-first.

    Replicates VecFrameStack: zero-padded at the episode start, newest
    frame last, then transposed for the policy.
    """
    T, H, W, C = obs.shape
    stacked = np.zeros((T, H, W, N_STACK * C), dtype=np.uint8)
    for t in range(T):
        for k in range(N_STACK):
            src = t - (N_STACK - 1 - k)  # k=N_STACK-1 is the current frame
            if src >= 0:
                stacked[t, :, :, k * C : (k + 1) * C] = obs[src]
    return np.transpose(stacked, (0, 3, 1, 2))


def episode_returns(n_steps: int, outcome: str, final_time: float,
                    armed_index: int = -1) -> np.ndarray:
    """Discounted returns under config.py's reward scheme.

    Mirrors env.py step-for-step so the pretrained value head matches what
    PPO will actually see:
      * -TIME_PENALTY_PER_STEP every step
      * +TIMER_RUNNING_REWARD_PER_STEP (capped per episode) on each ARMED,
        non-terminal step — the timer-running survival reward. armed_index
        is the step the timer started (saved in each demo); -1 means it
        never armed, so no survival reward at all.
      * the outcome bonus (finish/fall) on the terminal step, which in
        env.py takes the place of that step's survival reward.
    The optical-flow shaping is intentionally NOT modelled here: it's tiny,
    noisy, and can't be recomputed from metadata alone.
    """
    rewards = np.full(n_steps, -config.TIME_PENALTY_PER_STEP, dtype=np.float32)

    # A finish/fall/never_started demo ends ON its terminal step, which gets
    # the outcome bonus instead of a survival reward (env.py skips the else
    # branch on terminal steps). A timeout demo has no terminal event, so its
    # last step still earns survival reward.
    has_terminal_event = outcome in ("finish", "fell", "never_started")
    survival_end = n_steps - 1 if has_terminal_event else n_steps

    if armed_index is not None and armed_index >= 0:
        accum = 0.0
        for i in range(armed_index, survival_end):
            bonus = min(
                config.TIMER_RUNNING_REWARD_PER_STEP,
                max(0.0, config.TIMER_RUNNING_REWARD_MAX - accum),
            )
            rewards[i] += bonus
            accum += bonus

    if outcome == "finish":
        t = final_time if final_time and final_time > 0 else n_steps * config.STEP_DT
        rewards[-1] += max(
            config.FINISH_REWARD - t * config.FINISH_TIME_BONUS_SCALE,
            config.FINISH_MIN_REWARD,
        )
    elif outcome in ("fell", "never_started"):
        rewards[-1] += config.FALL_PENALTY

    returns = np.zeros(n_steps, dtype=np.float32)
    running = 0.0
    for i in range(n_steps - 1, -1, -1):
        running = rewards[i] + config.GAMMA * running
        returns[i] = running
    return returns


def load_demos(demo_dir: str):
    paths = sorted(glob.glob(os.path.join(demo_dir, "*.npz")))
    if not paths:
        raise SystemExit(f"No .npz demos found in {demo_dir}")

    episodes = []
    for path in paths:
        data = np.load(path, allow_pickle=True)
        obs = data["obs"]
        actions = data["actions"]
        outcome = str(data["outcome"])
        final_time = float(data["final_time"])
        # Step the timer started; needed to place the survival reward. Older
        # demos without it fall back to -1 (no survival reward modelled).
        armed_index = int(data["armed_index"]) if "armed_index" in data else -1

        if obs.shape[1:] != (*config.FRAME_SIZE, config.OBS_CHANNELS):
            print(f"  SKIP {os.path.basename(path)}: obs shape {obs.shape[1:]} "
                  f"doesn't match current config {(*config.FRAME_SIZE, config.OBS_CHANNELS)}")
            continue
        n_x, n_y = len(config.LOOK_DELTAS_X), len(config.LOOK_DELTAS_Y)
        if actions[:, 1].max() >= n_x or actions[:, 2].max() >= n_y:
            print(f"  SKIP {os.path.basename(path)}: action bins exceed current "
                  f"config — recorded with different LOOK_DELTAS. Re-record.")
            continue

        episodes.append({
            "obs": stack_episode(obs),
            "actions": actions,
            "returns": episode_returns(len(actions), outcome, final_time, armed_index),
            "outcome": outcome,
            "name": os.path.basename(path),
        })
    if not episodes:
        raise SystemExit("No usable demos after filtering.")
    return episodes


def build_arrays(episodes):
    obs = np.concatenate([e["obs"] for e in episodes], axis=0)
    actions = np.concatenate([e["actions"] for e in episodes], axis=0)
    returns = np.concatenate([e["returns"] for e in episodes], axis=0)
    return obs, actions, returns


def evaluate(model, obs, actions, device, batch_size):
    """Per-head accuracy of the greedy action against the demonstrator's."""
    correct = np.zeros(3)
    total = 0
    model.policy.set_training_mode(False)
    with torch.no_grad():
        for start in range(0, len(obs), batch_size):
            ob = torch.as_tensor(obs[start : start + batch_size]).to(device)
            ac = torch.as_tensor(actions[start : start + batch_size]).to(device)
            dist = model.policy.get_distribution(ob)
            for head in range(3):
                pred = dist.distribution[head].probs.argmax(dim=-1)
                correct[head] += (pred == ac[:, head]).sum().item()
            total += len(ob)
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Behavior-clone a PPO policy from human demos.")
    parser.add_argument("--demos", default="demos/run", help="Directory of .npz demo files.")
    parser.add_argument("--out", default="checkpoints/bc_pretrained.zip", help="Where to save the PPO .zip.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="Fraction of EPISODES (not frames) held out for validation.")
    parser.add_argument("--jump-weight", type=float, default=None,
                        help="Positive-class weight for the jump head. Default: inverse "
                             "frequency, capped at 20.")
    parser.add_argument("--value-coef", type=float, default=0.5,
                        help="Weight on value-head regression. 0 disables value pretraining.")
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"Loading demos from {args.demos}/")
    episodes = load_demos(args.demos)
    order = rng.permutation(len(episodes))
    n_val = max(1, int(len(episodes) * args.val_fraction)) if len(episodes) > 2 else 0
    val_eps = [episodes[i] for i in order[:n_val]]
    train_eps = [episodes[i] for i in order[n_val:]]

    train_obs, train_acts, train_rets = build_arrays(train_eps)
    print(f"  {len(episodes)} episodes ({len(train_eps)} train / {len(val_eps)} val), "
          f"{len(train_obs)} training frames")
    outcomes = {}
    for e in episodes:
        outcomes[e["outcome"]] = outcomes.get(e["outcome"], 0) + 1
    print(f"  outcomes: {outcomes}")

    jump_rate = float(train_acts[:, 0].mean())
    print(f"  jump rate in demos: {jump_rate:.1%}")
    if jump_rate < 0.005:
        print("  WARNING: almost no jumps recorded — check that the event tap saw "
              "SPACE (run record_demos.py --check-input).")

    if args.jump_weight is not None:
        jump_weight = args.jump_weight
    else:
        jump_weight = min(20.0, (1 - jump_rate) / max(jump_rate, 1e-6))
    print(f"  jump positive-class weight: {jump_weight:.2f}")

    vec_env = VecFrameStack(DummyVecEnv([lambda: SpecEnv()]), n_stack=N_STACK)
    model = PPO(
        "CnnPolicy",
        vec_env,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        gamma=config.GAMMA,
        ent_coef=config.ENT_COEF,
        verbose=0,
        device=args.device,
    )
    device = model.device
    print(f"  device: {device}")

    # Sanity check: the shape we built must match what the policy expects.
    expected = tuple(model.policy.observation_space.shape)
    if train_obs.shape[1:] != expected:
        raise SystemExit(f"Stacked obs {train_obs.shape[1:]} != policy input {expected}")

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=args.lr)
    jump_class_weights = torch.tensor([1.0, jump_weight], dtype=torch.float32, device=device)

    n = len(train_obs)
    for epoch in range(args.epochs):
        model.policy.set_training_mode(True)
        perm = rng.permutation(n)
        totals = np.zeros(4)  # policy loss, value loss, jump loss, batches

        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            obs_b = torch.as_tensor(train_obs[idx]).to(device)
            act_b = torch.as_tensor(train_acts[idx]).to(device)
            ret_b = torch.as_tensor(train_rets[idx]).to(device)

            dist = model.policy.get_distribution(obs_b)
            heads = dist.distribution

            # Jump head: weighted CE (jumps are rare; unweighted CE learns
            # "never jump" and calls it a day).
            jump_loss = F.cross_entropy(
                heads[0].logits, act_b[:, 0], weight=jump_class_weights
            )
            look_h_loss = F.cross_entropy(heads[1].logits, act_b[:, 1])
            look_v_loss = F.cross_entropy(heads[2].logits, act_b[:, 2])
            policy_loss = jump_loss + look_h_loss + look_v_loss

            loss = policy_loss
            value_loss_item = 0.0
            if args.value_coef > 0:
                values = model.policy.predict_values(obs_b).squeeze(-1)
                value_loss = F.mse_loss(values, ret_b)
                loss = loss + args.value_coef * value_loss
                value_loss_item = value_loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            optimizer.step()

            totals += [policy_loss.item(), value_loss_item, jump_loss.item(), 1]

        batches = max(totals[3], 1)
        msg = (f"epoch {epoch + 1:2d}/{args.epochs}  policy {totals[0] / batches:.4f}  "
               f"jump {totals[2] / batches:.4f}  value {totals[1] / batches:.4f}")
        if val_eps:
            val_obs, val_acts, _ = build_arrays(val_eps)
            acc = evaluate(model, val_obs, val_acts, device, args.batch_size)
            msg += f"   val acc  jump {acc[0]:.2f}  look_h {acc[1]:.2f}  look_v {acc[2]:.2f}"
        print(msg)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    model.save(args.out)
    print(f"\nSaved {args.out}")
    print("Fine-tune with:")
    print(f"  python train.py --name my_segment --resume-from {args.out} --timesteps 50000")
    print("Lower config.LEARNING_RATE (5e-5 to 1e-4) for the first fine-tuning run "
          "so PPO doesn't immediately undo the cloning.")


if __name__ == "__main__":
    main()
