import numpy as np
import torch
import torch.nn as nn
from Juggler_gym import Juggler 
from stable_baselines3 import PPO


class RandomThrowAgent():
    """
    Accelerate and open hand at random time.
    """
    def __init__(self, pattern):
        self.p_open = 0.01 # probability of open hand
        self.acc_bias = 3 # positive acceleration bias

    def act(self, observations, info):
        l_open = np.random.rand() < self.p_open
        r_open = np.random.rand() < self.p_open
        l_acc = np.random.randn() + self.acc_bias
        r_acc = np.random.randn() + self.acc_bias
        return np.array([not l_open, not r_open, l_acc, r_acc])


class PredictiveAgent(RandomThrowAgent):
    """
    An agent that learns to predict the trajectories of its throws.
    Concretely, it learns a function from ball position and velocity to ball position at the peak of the throw.
    The inverse difference of predicted peak to checkpoint can serve as a reward proxy.
    This way, we solve the problem of delayed rewards and causal credit assignment.
    """
    def __init__(self, pattern):
        self.pattern = pattern # TODO
        self.n_balls = sum(pattern) // len(pattern)

        # neural network for predicting peak position
        self.peak_model = nn.Sequential(
            nn.Linear(4, 10), # ball position and velocity
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 2) # ball position at peak of throw
        )
        self.optimizer = torch.optim.SGD(self.peak_model.parameters(), lr=0.001, momentum=0.9)
        self.episode_loss = [] # stores loss per episode
        self.reset_buffer()
        # saves trajectories, one per ball
        super().__init__(pattern)


    def reset_buffer(self):
        self.trajectory_buffer = [[]] * self.n_balls # TODO capacity?
        

    def act(self, observations, info):
        balls_obs = observations[6:]
        for b in range(self.n_balls):
            ball_state = balls_obs[b*4:(b+1)*4] # 4 variables per ball: pos (x,y) and vel (x,y)
            self.trajectory_buffer[b] += [ball_state] # add it to the buffer
        return super().act(observations, info)


    def learn_trajectories(self):
        # if episode terminated, train model
        losses = [] # for all balls
        for trajectory in self.trajectory_buffer:
            peak = max(trajectory, key=lambda x: x[1]) # get state with highest y value
            peak = torch.Tensor([peak[:2]] * len(trajectory)) # only position, not velocity
            data = torch.Tensor(trajectory)
            self.optimizer.zero_grad()
            pred = self.peak_model(data)
            loss = nn.MSELoss().forward(pred, peak)
            loss.backward()
            self.optimizer.step()
            losses += [loss]
        self.episode_loss += [sum(losses) / len(losses)]
        print(losses) # TODO
        # empty trajectory buffer
        self.reset_buffer()


    def predictive_reward(self, ball):
        state = [ball["pos"][0], ball["pos"][1], ball["vel"][0], ball["vel"][1]]
        pred = self.peak_model(state)
        apex = ball["apex"]
        return 1 / np.linalg.norm(pred - apex)



if __name__ == "__main__":
    PATTERN = [3,3,0] # [4,4,4,4]
    N_STEP = 1000

    env = Juggler(PATTERN, render_mode=None, verbose=False)
    agent = PredictiveAgent(PATTERN)

    for epi in range(1000):
        terminate = False
        rewards = []
        step = 0
        obs, info = env.reset()
        while not terminate and step < N_STEP:
            ctrl = agent.act(obs, info)
            obs, reward, terminate, _, info = env.step(ctrl)
            rewards += [reward]
            #print(reward)
            env.render()
            step += 1
        agent.learn_trajectories()