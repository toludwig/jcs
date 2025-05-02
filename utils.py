import torch
import numpy as np
from collections import deque



def get_dmc_state(env, time_step):
    state = torch.from_numpy(np.hstack([arr.ravel() for arr in time_step.observation.values()])).unsqueeze(0)
    return state


def compute_num_features(env):
    time_step = env.reset()
    state = get_dmc_state(env, time_step)
    return state.size(1)

def env_step_repeat(env, action, n=1):
    reward = 0
    for i in range(n):
        time_step = env.step(action)
        reward += time_step.reward
        done = time_step.last()
        if done:
            break
    return time_step, reward

class FrameStacker:
    def __init__(self, num_stacked=3, channels=3):
        self.num_stacked = num_stacked
        self.channels = channels
        self.frames = deque(maxlen=num_stacked)

    def reset(self):
        self.frames.clear()

    def get_dmc_pixels(self, env, time_step):
        frame = env.physics.render(camera_id=0)  #
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        self.frames.append(frame_tensor)
        while len(self.frames) < self.num_stacked:
            self.frames.append(frame_tensor.clone())
        return torch.cat(list(self.frames), dim=0)

def run_test_episodes(env, agent, repeats, num_episodes=10, pixels = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pixels:
        frame_stacker = FrameStacker(num_stacked=3, channels=3)
        get_state_function = frame_stacker.get_dmc_pixels
    else:
        get_state_function = get_dmc_state
    with torch.no_grad():
        episode_length = int(1000/repeats)
        rewards = torch.zeros(num_episodes, episode_length)
        for i in range(num_episodes):
            time_step = env.reset()
            if pixels:
                frame_stacker.reset()
            current_state =  get_state_function(env, time_step)
            step = 0
            while not time_step.last():
                action = agent.act_deterministic(current_state.float().to(device=device)).cpu().detach().numpy()
                #action = agent.step_deterministic(current_state.float().to(device=device)).cpu().detach().numpy()
                time_step, reward = env_step_repeat(env, action, n=repeats)
                rewards[i, step] = reward#time_step.reward
                next_state = get_state_function(env, time_step)#torch.from_numpy(np.hstack(list(time_step.observation.values())))#torch.cat(tuple(torch.tensor(val).view(-1, 1) for val in time_step.observation.values())).T.squeeze(0)
                current_state = next_state
                step += 1
        return rewards.sum(dim=1)



