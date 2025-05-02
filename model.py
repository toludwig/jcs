# code adapted from https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/model.py

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli, Categorical
import numpy as np



def weight_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data, gain=1)
        m.bias.data.fill_(0.01)

class BernoulliActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim= 256):
        super(BernoulliActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.action_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(weight_init_)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.action_linear(x))
        return Categorical(probs=x)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(GaussianPolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(weight_init_)
        self.log_std_bounds = [-10, 2]


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = F.tanh(log_std)

        log_std = self.log_std_bounds[0] +  log_std * (self.log_std_bounds[1] - self.log_std_bounds[0]) / 2

        policy = Normal(mean, torch.exp(log_std))
        return policy

class MixedPolicyActor(nn.Module):
    def __init__(self, state_dim, num_binary_actions=2, num_continuous_actions=2, hidden_dim=256):
        super(MixedPolicyActor, self).__init__()

        self.num_binary = num_binary_actions
        self.num_cont = num_continuous_actions

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # For Bernoulli actions (binary 0/1)
        self.binary_head = nn.Linear(hidden_dim, num_binary_actions)

        # For Gaussian actions (continuous)
        self.cont_mean_head = nn.Linear(hidden_dim, num_continuous_actions)
        self.cont_log_std_head = nn.Linear(hidden_dim, num_continuous_actions)

        self.log_std_bounds = [-10, 2]

        self.apply(weight_init_)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Bernoulli distribution for discrete actions
        binary_probs = torch.sigmoid(self.binary_head(x))
        binary_dist = Bernoulli(probs=binary_probs)

        # Gaussian distribution for continuous actions
        mean = self.cont_mean_head(x)
        log_std = self.cont_log_std_head(x)
        log_std = F.tanh(log_std)
        log_std = self.log_std_bounds[0] + log_std * (self.log_std_bounds[1] - self.log_std_bounds[0]) / 2
        std = torch.exp(log_std)

        cont_dist = Normal(mean, std)

        return binary_dist, cont_dist


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        self.apply(weight_init_)

    def forward(self, x, action):

        x = torch.cat([x, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.value_head(x)


# class SAC(nn.Module):
#     def __init__(self, state_dims, binary_actions=2, continuous_actions=2, hidden_dims=256, gamma=0.99, alpha=0.1, rho=0.01):
#         super(SAC, self).__init__()
#         self.state_dims = state_dims
#         self.action_dims = binary_actions + continuous_actions
#         self.gamma = gamma
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#         self.actor = MixedPolicyActor(self.state_dims, num_continuous_actions=continuous_actions, num_binary_actions=binary_actions)
#
#         # Q Networks
#         self.q1 = Critic(state_dims, self.action_dims, hidden_dims).to(self.device)
#         self.q2 = Critic(state_dims, self.action_dims, hidden_dims).to(self.device)
#
#         # Target Q Networks
#         self.q_target1 = Critic(state_dims, self.action_dims, hidden_dims).to(self.device)
#         self.q_target2 = Critic(state_dims, self.action_dims, hidden_dims).to(self.device)
#         self.src_models = [self.q1, self.q2]
#
#         # Initialize target networks with the same weights as the main Q-networks
#         self.q_target1.load_state_dict(self.q1.state_dict())
#         self.q_target2.load_state_dict(self.q2.state_dict())
#         self.target_models = [self.q_target1, self.q_target2]
#
#         self.target_entropy = -continuous_actions  - binary_actions*np.log(binary_actions)
#
#         self.log_alpha = torch.nn.Parameter(torch.tensor([alpha], device="cuda" if torch.cuda.is_available() else "cpu").log())
#         self.rho = rho
#
#     def act(self, state):
#         binary_dist, cont_dist = self.actor(state)
#         binary_action = binary_dist.sample()
#         cont_action = cont_dist.sample()
#         cont_action = torch.tanh(cont_action)
#         action = torch.cat([binary_action, cont_action], dim=-1)
#         return action
#
#     def act_deterministic(self, state):
#         binary_dist, cont_dist = self.actor(state)
#         binary_action = binary_dist.mode
#         cont_action = cont_dist.mean
#         cont_action = torch.tanh(cont_action)
#         action = torch.cat([binary_action, cont_action], dim=-1)
#         return action
#
#     def get_action_prob(self,s):
#         binary_dist, cont_dist = self.actor(s)
#
#         # ----- Binary action sampling -----
#         binary_action = binary_dist.sample()  # Sample binary action (e.g., 0 or 1)
#         binary_log_prob = binary_dist.log_prob(binary_action)  # Log-probability of binary action
#         binary_log_prob = binary_log_prob.sum(1, keepdim=True)
#
#         # ----- Continuous action sampling -----
#         action = cont_dist.rsample()  # Differentiable sample
#         action_t = torch.tanh(action)  # Squash to [-1, 1]
#         log_probs = cont_dist.log_prob(action)  # Log-prob before squashing
#         log_probs -= torch.log(1 - action_t.pow(2) + 1e-6)  # Correction for tanh squashing
#         log_probs = log_probs.sum(1, keepdim=True)
#
#         # ----- Combine binary and continuous actions -----
#         combined_actions = torch.cat([binary_action, action_t], dim=-1)
#         combined_log_probs = torch.cat([binary_log_prob, log_probs], dim=-1)
#         print("combined_log_probs",combined_log_probs)
#         combined_log_probs = combined_log_probs.sum(1, keepdim=True)
#
#         return combined_actions, combined_log_probs
#
#     def get_q_vals(self, s, a, target=True):
#         if target:
#             q_vals1 = self.q_target1(s, a)
#             q_vals2 = self.q_target2(s, a)
#             q_vals = torch.min(q_vals1, q_vals2)
#         else:
#             q_vals1 = self.q1(s, a)
#             q_vals2 = self.q2(s, a)
#             q_vals = torch.min(q_vals1, q_vals2)
#         return q_vals
#
#
#     def train_critic(self, buffer):
#         s, a, s_prime, r, terminal = buffer.sample()
#         self.s = s
#
#         with torch.no_grad():
#             # simulate next action using policy
#             action_next, log_probs = self.get_action_prob(s_prime)
#             q_prime = self.get_q_vals(s_prime, action_next, target=True)  # get Q values for next state and action
#             q_prime = q_prime - (self.log_alpha.exp().detach() * log_probs)
#             target = r + ((self.gamma * (1 - terminal)) * q_prime)  # compute target
#
#         # train networks to predict target
#         q_vals1 = self.q1(s, a)
#         q_vals2 = self.q2(s, a)
#
#         crit = nn.MSELoss()
#         loss1 = crit(q_vals1, target)
#         loss2 = crit(q_vals2, target)
#
#         return loss1 + loss2
#
#     def train_actor(self):
#         s = self.s
#         a, log_probs = self.get_action_prob(s)
#         q_prime = self.get_q_vals(s, a, target=False)
#         loss = ((self.log_alpha.exp().detach() * log_probs) - q_prime).mean()
#
#         return loss
#
#     def train_actor_and_alpha(self):
#         s = self.s
#         a, log_probs = self.get_action_prob(s)  # get all action probabilities and log probs
#         q_prime = self.get_q_vals(s, a, target=False)
#         loss = ((self.log_alpha.exp().detach() * log_probs) - q_prime).mean()
#         alpha_loss = (self.log_alpha.exp() * (-log_probs - self.target_entropy).detach()).mean()
#         print("shape of log probs",log_probs.shape)
#         print("log probs",log_probs)
#         # alpha_loss = (-self.alpha.exp() * (log_probs - (-self.target_entropy)).detach()).mean()
#         return loss, alpha_loss
#
#     def soft_update(self):
#         """Updates the target network in the direction of the local network but by taking a step sizeg"""
#         for (target_model, local_model) in zip(self.target_models, self.src_models):
#
#             for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
#                 target_param.data.copy_((self.rho * local_param.data) + ((1.0 - self.rho) * target_param.data))
#


class ActorContinuous(nn.Module):
    def __init__(self, input_dims, action_dims, hidden_dims=400):
        super(ActorContinuous, self).__init__()

        self.input_dims = input_dims
        self.action_dims = action_dims
        self.encoder = nn.Sequential(
                        nn.Linear(input_dims, hidden_dims),
                        nn.ReLU(),
                        nn.Linear(hidden_dims, hidden_dims),
                        nn.ReLU(),
                        nn.Linear(hidden_dims, action_dims*2)
        )
        self.apply(weight_init_)
        self.log_std_bounds = [-10, 2]

    def forward(self, s):

        m, log_std = self.encoder(s).chunk(2, dim=-1)

        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +1)

        sd = log_std.exp()
        return m, sd


class IdentityEncoder(nn.Module):
    def __init__(self):
        super(IdentityEncoder, self).__init__()
        self.im_size = 0

    def forward(self, x):
        return x

class Actor(nn.Module):
    def __init__(self, state_dims, action_dims, hidden_dims=256):
        super(Actor, self).__init__()
        self.action_dims = action_dims

        self.policy_head = nn.Sequential(
                        nn.Linear(state_dims, hidden_dims),
                        nn.ReLU(),
                        nn.Linear(hidden_dims, hidden_dims),
                        nn.ReLU(),
                        nn.Linear(hidden_dims, action_dims),
                        nn.Softmax()
        )
        self.apply(weight_init_)

    def forward(self, s):
        return self.policy_head(s)


class SAC(nn.Module):
    def __init__(self, state_dims, action_dims, hidden_dims=256, gamma=0.99, alpha=0.1, rho=0.01):
        super(SAC, self).__init__()
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = IdentityEncoder()


        self.policy_head = Actor(state_dims, action_dims, hidden_dims).to(self.device)

        self.qtarget1 = Critic(state_dims, action_dims, hidden_dims).to(self.device)
        self.qtarget2 = Critic(state_dims, action_dims, hidden_dims).to(self.device)

        self.qsrc1 = Critic(state_dims, action_dims, hidden_dims).to(self.device)
        self.qsrc2 = Critic(state_dims, action_dims, hidden_dims).to(self.device)

        self.target_models = [self.qtarget1, self.qtarget2]
        self.src_models = [self.qsrc1, self.qsrc2]

        self.log_alpha = torch.nn.Parameter(torch.tensor([alpha], device="cuda" if torch.cuda.is_available() else "cpu").log())
        self.rho = rho
        self.action_indices = torch.arange(self.action_dims).to(device=self.device)
        self.action_vectors = torch.eye(self.action_dims).float().to(device=self.device)

    def forward(self, state):
        p = self.policy_head(state)
        policy = torch.distributions.Categorical(p)
        return policy

    def act(self, s):

        s = self.encoder(s)
        policy = self(s)
        return policy.sample()

    def act_deterministic(self, s):

        s = self.encoder(s)
        policy = self(s)
        return policy.probs.argmax(dim=-1)


    def concatenate_actions(self, s):
        '''Takes a tensor of embedding vectors and adds all actions for every vector'''
        s_rep = torch.repeat_interleave(s, self.action_dims, dim=0)
        a_rep = self.action_vectors.repeat(s.shape[0], 1)

        return torch.cat((s_rep, a_rep), dim=-1), s_rep.to(device=self.device), a_rep.to(device=self.device)

    def get_action_probabilities(self, s):
        policy = self(s)
        #actions = policy.sample()
        action_probabilities = policy.probs
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_probs = torch.log(action_probabilities + z)

        return action_probabilities, log_probs


    def get_q_vals(self, s, target = True):
        sa, s_repeat, a_repeat = self.concatenate_actions(s)

        if target:
            q_vals1 = self.qtarget1(s_repeat, a_repeat).reshape(s.size(0), self.action_dims)
            q_vals2 = self.qtarget2(s_repeat, a_repeat).reshape(s.size(0), self.action_dims)
            q_vals = torch.min(q_vals1, q_vals2)
        else:
            q_vals1 = self.qsrc1(s_repeat, a_repeat).reshape(s.size(0), self.action_dims)
            q_vals2 = self.qsrc2(s_repeat, a_repeat).reshape(s.size(0), self.action_dims)
            q_vals = torch.min(q_vals1, q_vals2)
        return q_vals

    def train_critic(self, buffer):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        s, a, s_prime, r, terminal = buffer.sample()
        self.s = s
        s = self.encoder(s)
        #self._s_rep = s.detach()
        s_prime = self.encoder(s_prime)
        with torch.no_grad():


            action_probs, log_probs = self.get_action_probabilities(s_prime) # get all action probabilities and log probs
            q_prime = self.get_q_vals(s_prime, target = True)    # get all Q values
            q_prime = action_probs * (q_prime - self.log_alpha.exp().detach() * log_probs)   # compute expectation by weighing according to p
            q_prime = q_prime.sum(dim=1).unsqueeze(-1)  # integrate
            target = r + ((self.gamma*(1-terminal))*q_prime)


        #sa = torch.cat((s, a), dim=-1)
        q_vals1 = self.qsrc1(s, a)
        q_vals2 = self.qsrc2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)

        return loss1 + loss2


    def train_actor_and_alpha(self):
        s = self.encoder(self.s)
        action_probs, log_probs = self.get_action_probabilities(s) # get all action probabilities and log probs
        q_prime = self.get_q_vals(s, target = False)
        inside_term = self.log_alpha.exp().detach()*log_probs - q_prime
        loss = (action_probs*inside_term).sum(dim=1).mean()

        return loss

    def soft_update(self):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for (target_model, local_model) in zip(self.target_models, self.src_models):

            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_((self.rho*local_param.data) + ((1.0-self.rho)*target_param.data))


class SACContinuous(SAC):

    def __init__(self, state_dims, action_dims, hidden_dims=256, gamma=0.99, alpha=0.1, rho=0.01):
        super(SACContinuous, self).__init__(state_dims, action_dims, hidden_dims, gamma, alpha, rho)
        self.target_entropy = -action_dims
        self.policy_head = ActorContinuous(state_dims, action_dims, hidden_dims).to(self.device)#nn.Linear(self.state_dims, self.action_dims)

    def forward(self, state):
        mu, sd = self.policy_head(state)
        policy = Normal(mu, sd)
        return policy


    def act(self, s):

        s = self.encoder(s)
        policy = self(s)
        return torch.tanh(policy.sample())

    def act_deterministic(self, s):

        s = self.encoder(s)
        policy = self(s)
        return torch.tanh(policy.mean)


    def get_action_probabilities(self, s):
        policy = self(s)
        action = policy.rsample()
        action_t = torch.tanh(action) # squish
        log_probs = policy.log_prob(action)#.sum(dim=-1, keepdim=True)#.sum(1, keepdim=True)
        # # apply tanh squishing of log probs
        log_probs -= torch.log(1 - action_t.pow(2)+1e-6)
        log_probs = log_probs.sum(1, keepdim=True)
        #action_probs = torch.exp(log_probs)#policy.probs(action)
        #return action, log_probs
        return action_t, log_probs, action



    def get_q_vals(self, s, a, target = True):


        if target:
            q_vals1 = self.qtarget1(s, a)
            q_vals2 = self.qtarget2(s, a)
            q_vals = torch.min(q_vals1, q_vals2)
        else:
            q_vals1 = self.qsrc1(s, a)
            q_vals2 = self.qsrc2(s, a)
            q_vals = torch.min(q_vals1, q_vals2)
        return q_vals

    def train_critic(self, buffer):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        s, a, s_prime, r, terminal = buffer.sample()
        self.s = s

        s = self.encoder(s)
        s_prime = self.encoder(s_prime)

        with torch.no_grad():
            # simulate next action using policy
            action_next, log_probs, _ = self.get_action_probabilities(s_prime)
            q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action
            q_prime = q_prime - (self.log_alpha.exp().detach()*log_probs)
            target = r + ((self.gamma*(1-terminal))*q_prime) # compute target


        # train networks to predict target
        q_vals1 = self.qsrc1(s, a)
        q_vals2 = self.qsrc2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)

        return loss1 + loss2


    def train_actor_and_alpha(self):
        s = self.encoder(self.s)
        a, log_probs, _ = self.get_action_probabilities(s) # get all action probabilities and log probs
        q_prime = self.get_q_vals(s, a, target = False)
        loss = ((self.log_alpha.exp().detach() * log_probs) - q_prime).mean()
        alpha_loss = (self.log_alpha.exp() * (-log_probs - self.target_entropy).detach()).mean()
        return loss, alpha_loss


if __name__ == '__main__':
    state_dims = 3
    action_dims = 4
    actor = MixedPolicyActor(state_dims)
    critic = Critic(state_dims, action_dims)

    policy_d, policy_con = actor(torch.tensor(np.random.randn(1, state_dims)).float())

    print(policy_con.sample(), policy_d.sample())

    print(critic(torch.tensor(np.random.randn(1, state_dims)).float(), torch.tensor(np.random.randn(1, action_dims)).float()))
