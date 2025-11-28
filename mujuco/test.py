import gymnasium as gym
import numpy as np

env = gym.make("Humanoid-v5", render_mode="human")  # or "human" / "rgb_array"
obs, info = env.reset(seed=42)

for step in range(1000):
    action = env.action_space.sample()           # shape ~ (17,)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()