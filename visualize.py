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
from point_vessel_env import PointVesselEnv

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None  # allow random-demo without SB3

# ---------------------------------------------------------------------

def run_episode(env: PointVesselEnv, model, max_steps=3000):
    obs, _ = env.reset()
    done = truncated = False
    ep_rew = 0.0
    while not (done or truncated):
        # check for pygame quit
        import pygame
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         env.close(); sys.exit(0)

        # choose action
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, r, done, truncated, info = env.step(action)
        env.render()
        ep_rew += r
        time.sleep(1 / env.metadata["render_fps"])
        if env.steps > max_steps:
            break
    print(f"Reward {ep_rew:.1f} | success={info['success']} | "
          f"dist={info['distance']:.2f} | collided={info['collided']}")
    time.sleep(1)  # pause before next episode

def main():
    parser = argparse.ArgumentParser(
        description="Visualise catheter RL agent")
    parser.add_argument("model", nargs="?", default=None,
                        help="Path to PPO .zip checkpoint (optional)")
    parser.add_argument("--episodes", type=int, default=1, help="# episodes")
    parser.add_argument("--save_frames", type=bool, default=False)
    args = parser.parse_args()

    use_model = args.model and Path(args.model).is_file()
    if use_model and PPO is None:
        sys.exit("Stable-Baselines3 not installed – `pip install stable-baselines3[extra]`")

    # env = Cath2DEnv(render_mode="human")
    env = PointVesselEnv(render_mode="human", save_frames=args.save_frames)
    if use_model:
        model = PPO.load(args.model, env=env, device="cpu")
        print(f"Loaded model {args.model}")
    else:
        model = None
        print("Running with RANDOM actions (no model supplied)")

    try:
        # for ep in range(args.episodes):
        #     obs, _ = env.reset(seed=ep)
        #     done = truncated = False
        #     ep_rew, steps = 0, 0
        #     while not (done or truncated):
        #         action = (model.predict(obs, deterministic=True)[0] if model
        #                   else env.action_space.sample())
        #         obs, rew, done, truncated, info = env.step(action)
        #         env.render()
        #         ep_rew += rew; steps += 1
        #         # small delay for human-eye framerate (if needed)
        #         time.sleep(1 / env.metadata["render_fps"])
        #     print(f"Ep {ep+1}/{args.episodes}: reward {ep_rew:.1f} | "
        #           f"{'SUCCESS' if info.get('success') else 'fail'} | "
        #           f"tip err {info.get('tip_error_mm', np.nan):.2f} mm")
        #       time.sleep(1.0)

        for ep in range(args.episodes):
            print(f"\nEpisode {ep+1}/{args.episodes}")
            run_episode(env, model)
            # let viewer settle

    finally:
        env.close()

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
