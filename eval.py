import numpy as np
from stable_baselines3 import PPO
from catheter_env import Cath2DEnv

env = Cath2DEnv(render_mode=None)
model = PPO.load("ppo_cath_tip")
success, tip_err, collisions = 0, [], 0

for ep in range(100):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, trunc, info = env.step(action)
    success += info["success"]
    tip_err.append(info["tip_error_mm"])
    collisions += info["collisions"]

print(f"Success {success} %, tip err {np.mean(tip_err):.2f} mm, "
      f"collisions {collisions/100:.2f} per episode")
