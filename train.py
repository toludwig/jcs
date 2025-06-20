import torch

from juggler import Juggler
from model import SAC, SACContinuous
from torch import nn, optim
import random
import numpy as np
from buffer import Buffer
import wandb


def evaluate(test_env, agent, device, train_episode, num_episodes=1):
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_reward = 0
        state = test_env.reset(rendering=True)
        done = False
        while not done:
            action = agent.act_deterministic(torch.tensor(state).float().to(device)).detach().cpu().numpy()
            next_state, reward, done = test_env.step(action)
            episode_reward += reward
            state = next_state
        all_episode_rewards.append(episode_reward)

    test_env.render("./render/episode_" + str(train_episode))
    wandb.log({"example": wandb.Video("./render/episode_" + str(train_episode) + ".gif", format="gif"),
               "eval_reward": np.mean(all_episode_rewards)
               })


def train_sac(seed, run,args):
    env = Juggler(args.pattern, verbose=False)
    test_env = Juggler(args.pattern, verbose=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the buffer
    binary_action_dim = 2
    cont_action_dim = 2
    action_dim = binary_action_dim + cont_action_dim
    state_dim = env.state_dim

    agent = SACContinuous(state_dims=state_dim, action_dims=action_dim, hidden_dims=args.hidden_dims, gamma=0.99, alpha=args.alpha, rho=args.rho).to(device)

    optimizer_actor = optim.Adam(list(agent.policy_head.parameters()),lr=args.lr)
    optimizer_critics = optim.Adam(list(agent.qtarget1.parameters()) + list(agent.qtarget2.parameters()), lr=args.lr)
    optimizer_alpha = optim.Adam([agent.log_alpha], lr=args.lr)

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    num_episodes = 500
    episode_length = 1500
    buffer = Buffer(episode_length=episode_length, buffer_size=1000000, batch_size = args.batch_size)
    num_random = 500 # how many steps to take random actions

    for i in range(num_episodes):
        print(i)
        current_state = env.reset()
        current_state = torch.tensor(current_state).to(device)
        episode_s = torch.zeros(episode_length, state_dim)
        episode_a = torch.zeros(episode_length, action_dim)
        episode_s_prime = torch.zeros(episode_length, state_dim)
        episode_r = torch.zeros(episode_length, 1)
        episode_terminal = torch.zeros(episode_length, 1)
        episode_loss = 0
        episode_loss_alpha = 0
        episode_loss_crit = 0
        for step in range(episode_length):
            episode_s[step] = current_state
            if step < num_random:
                action = env.sample_action()
            else:
                action = agent.act(current_state.float().to(device)).detach().cpu().numpy()

            next_state, reward, drop = env.step(action)
            next_state = torch.tensor(next_state, device=device)
            episode_r[step] = reward
            episode_a[step] = torch.from_numpy(action)
            episode_s_prime[step] = next_state
            episode_terminal[step] = drop
            current_state = next_state
            if i >= 4: # warmup
                s, a, s_prime, r, terminal = buffer.sample()
                critic_loss = agent.train_critic(buffer)
                optimizer_critics.zero_grad()
                critic_loss.backward()
                optimizer_critics.step()
                if step % 2 == 0:
                    actor_loss, alpha_loss = agent.train_actor_and_alpha()
                    episode_loss += actor_loss
                    episode_loss_crit += critic_loss
                    episode_loss_alpha += alpha_loss

                    optimizer_actor.zero_grad()
                    actor_loss.backward()
                    optimizer_actor.step()

                    optimizer_alpha.zero_grad()
                    alpha_loss.backward()
                    optimizer_alpha.step()

                    agent.soft_update()

        run.log({"total_reward": episode_r.sum().item()})
        run.log({"episode_reward": episode_r})
        run.log({"actor_loss": episode_loss})
        run.log({"critic_loss": episode_loss_crit})
        run.log({"alpha_loss": episode_loss_alpha})
        #run.log({"episode": i})
        run.log({"alpha": agent.log_alpha.exp().item()})

        buffer.append(episode_s, episode_a, episode_s_prime, episode_r, episode_terminal)
        buffer.finish_episode()
        if i % 25 == 0:
            evaluate(test_env, agent, device, i)