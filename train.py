import torch

from juggler import Juggler
from model import SAC
from torch import nn, optim
import random
import numpy as np
from buffer import Buffer


def train_sac(seed, run,args):
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
        current_state = torch.tensor(current_state).to(device)
        episode_s = torch.zeros(episode_length, state_dim)
        episode_a = torch.zeros(episode_length, action_dim)
        episode_s_prime = torch.zeros(episode_length, state_dim)
        episode_r = torch.zeros(episode_length, 1)
        episode_terminal = torch.zeros(episode_length, 1)
        episode_loss = 0
        epsiode_loss_alpha = 0
        for step in range(episode_length):
            episode_s[step] = current_state
            if step < num_random:
                action = env.sample_action()
            else:
                action = agent.act(current_state.float().to(device)).detach().cpu().numpy()[0]

            next_state, reward, drop = env.step(action)
            next_state = torch.tensor(next_state, device=device)
            episode_r[step] = reward
            episode_a[step] = torch.from_numpy(action)
            episode_s_prime[step] = next_state
            episode_terminal[step] = drop
            current_state = next_state
            if i >= 4:
                s, a, s_prime, r, terminal = buffer.sample()
                print(a)
                print(s_prime.shape)
                print(s.shape)
                critic_loss = agent.train_critic(buffer)
                optimizer_critics.zero_grad()
                critic_loss.backward()
                optimizer_critics.step()
                if step % 2 == 0:
                    #    print('hello')
                    actor_loss, alpha_loss = agent.train_actor_and_alpha()
                    episode_loss += actor_loss
                    epsiode_loss_alpha += alpha_loss

                    optimizer_actor.zero_grad()
                    actor_loss.backward()
                    optimizer_actor.step()

                    agent.soft_update()

        run.log({"reward": episode_r.sum().item()})

        run.log({"actor_loss": episode_loss})
        run.log({"critic_loss": epsiode_loss_alpha})
        buffer.append(episode_s, episode_a, episode_s_prime, episode_r, episode_terminal)
        buffer.finish_episode()









