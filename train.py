import torch, gymnasium as gym
from stable_baselines3 import PPO

from point_vessel_env import PointVesselEnv
env = PointVesselEnv()

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(net_arch=[128, 128]),
    n_steps=2048,
    batch_size=256,
    learning_rate=3e-4,
    gamma=0.99,
    clip_range=0.2,
    tensorboard_log="runs/ppo_cath/",
    verbose=1,
)
model.learn(total_timesteps=500_000)
model.save("ppo_point")