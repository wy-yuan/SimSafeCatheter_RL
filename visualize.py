# Visualise a trained PPO catheter-steering agent (or a random agent) in
# the Cath2DEnv environment. Press ESC to quit early.
#
# Usage:
#   python visualize.py              # random actions
#   python visualize.py ppo_cath_tip # load SB3 .zip weights
#
# ---------------------------------------------------------------------
import sys, argparse, time
import numpy as np
from pathlib import Path

from catheter_env import Cath2DEnv

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None  # allow random-demo without SB3

# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Visualise catheter RL agent in Cath2DEnv")
    parser.add_argument("model", nargs="?", default=None,
                        help="Path to SB3 .zip checkpoint (optional)")
    parser.add_argument("--episodes", type=int, default=5, help="# episodes")
    args = parser.parse_args()

    use_model = args.model and Path(args.model).is_file()
    if use_model and PPO is None:
        sys.exit("Stable-Baselines3 not installed – `pip install stable-baselines3[extra]`")

    env = Cath2DEnv(render_mode="human")
    if use_model:
        model = PPO.load(args.model, env=env, device="cpu")
        print(f"Loaded model {args.model}")
    else:
        model = None
        print("Running with RANDOM actions (no model supplied)")

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=ep)
            done = truncated = False
            ep_rew, steps = 0, 0
            while not (done or truncated):
                action = (model.predict(obs, deterministic=True)[0] if model
                          else env.action_space.sample())
                obs, rew, done, truncated, info = env.step(action)
                env.render()
                ep_rew += rew; steps += 1
                # small delay for human-eye framerate (if needed)
                time.sleep(1 / env.metadata["render_fps"])
            print(f"Ep {ep+1}/{args.episodes}: reward {ep_rew:.1f} | "
                  f"{'SUCCESS' if info.get('success') else 'fail'} | "
                  f"tip err {info.get('tip_error_mm', np.nan):.2f} mm")
            # let viewer settle
            time.sleep(1.0)
    finally:
        env.close()

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
