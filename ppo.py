import gymnasium as gym
import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import BaseCallback
import wandb
from wandb.integration.sb3 import WandbCallback
from Juggler_gym import Juggler


class CustomWandbCallback(WandbCallback):
    """
    Custom callback that extends WandbCallback to log additional reward metrics
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        # Get the current reward from the environment
        if len(self.training_env.buf_rews) > 0:
            reward = self.training_env.buf_rews[0]
            self.current_episode_reward += reward
            self.current_episode_length += 1
            
            # Log step reward
            wandb.log({
                "step_reward": reward,
                "cumulative_episode_reward": self.current_episode_reward,
                "episode_length": self.current_episode_length
            })
        
        # Check if episode is done
        if len(self.training_env.buf_dones) > 0 and self.training_env.buf_dones[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Log episode statistics
            wandb.log({
                "episode_reward": self.current_episode_reward,
                "episode_length": self.current_episode_length,
                "mean_episode_reward": np.mean(self.episode_rewards),
                "std_episode_reward": np.std(self.episode_rewards),
                "min_episode_reward": np.min(self.episode_rewards),
                "max_episode_reward": np.max(self.episode_rewards),
                "mean_episode_length": np.mean(self.episode_lengths),
                "episode_count": len(self.episode_rewards)
            })
            
            # Reset episode counters
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        # Call parent method to maintain original functionality
        return super()._on_step()


if __name__ == "__main__":

    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 3*25000,
        "env_name": "Juggler-v0",
    }
    run = wandb.init(
        entity="tobiludw-university-t-bingen",
        project="JSC",
        name="test_experiment",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )


    def make_env():
        env = gym.make(config["env_name"], pattern= [3,0,0])
        env = Monitor(env)  # record stats such as returns
        return env


    env = DummyVecEnv([make_env])


    model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=CustomWandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    # for video in os.listdir(f"videos/{run.id}"):
    #     run.log({"video": wandb.Video(f"videos/{run.id}/{video}")})
    run.finish()