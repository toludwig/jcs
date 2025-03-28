import torch

from juggler import Juggler
from model import SAC
from torch import nn, optim
import random
import numpy as np
from buffer import Buffer


def train_sac(seed, args):
    pattern = [3, 3, 3]
    env = Juggler(pattern, rendering=False, verbose=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the buffer
    binary_action_dim = 2
    cont_action_dim = 2
    action_dim = binary_action_dim + cont_action_dim
    state_dim = env.state_dim

    agent = SAC(state_dims=state_dim, binary_actions=binary_action_dim,
                continuous_actions=cont_action_dim, hidden_dims=args.hidden_dims, gamma=0.99, alpha=args.alpha, rho=args.rho).to(device)

    optimizer_actor = optim.Adam(list(agent.actor.parameters()),lr=args.lr)
    optimizer_critics = optim.Adam(list(agent.q1.parameters()) + list(agent.q2.parameters()), lr=args.lr)

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    #env, mt1 = get_metaworld_env(args.task_name, seed)
    episode_length = 350
    buffer = Buffer(episode_length=episode_length, buffer_size=1000000, batch_size = args.batch_size)
    num_random = 1000


    for i in range(episode_length):
        current_state = env.reset()
        step = 0
        episode_s = torch.zeros(episode_length, state_dim)
        episode_a = torch.zeros(episode_length, action_dim)
        episode_s_prime = torch.zeros(episode_length, state_dim)
        episode_r = torch.zeros(episode_length, 1)
        terminal = torch.zeros(episode_length, 1)
        successful_task = False
        for step in range(episode_length):
            episode_s[step] = current_state
            if step < num_random:
                action = env.action_space.sample()
            else:
                action = agent.act(current_state.float().to(device)).detach().cpu().numpy()[0]

            next_state, reward, drop = env.step(*action)
            episode_r[step] = reward
            episode_a[step] = torch.from_numpy(action)
            episode_s_prime[step] = next_state
            current_state = next_state
            if i >= 4:
                critic_loss = agent.train_critic(buffer)
                optimizer_critics.zero_grad()
                critic_loss.backward()
                optimizer_critics.step()
                if step % 2 == 0:
                    #    print('hello')
                    actor_loss, alpha_loss = agent.train_actor()
                    optimizer_actor.zero_grad()
                    actor_loss.backward()
                    optimizer_actor.step()

                    agent.soft_update()

        buffer.append(episode_s, episode_a, episode_s_prime, episode_r, terminal)
        buffer.finish_episode()









