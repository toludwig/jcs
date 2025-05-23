import sys

import numpy as np
import wandb
from dm_control import suite
import argparse
import torch
from torch import nn, optim
from utils import *
from importlib import reload
from matplotlib import pyplot as plt
from model import SACContinuous
from buffer import Buffer

import random
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='Arguments for running experiments')
parser.add_argument('--num_seeds', metavar='N', type=int, default = 1,
                    help='number of seeds used')

parser.add_argument('--manual_seed', metavar='N', type=int, default = 0,
                    help='number of seeds used')


parser.add_argument('--update_freq', metavar='N', type=int, default = 2,
                    help='number of episodes in experiment')

parser.add_argument('--num_episodes', metavar='N', type=int, default = 160,
                    help='number of episodes in experiment')

parser.add_argument('--action_repeat', metavar='N', type=int, default = 4,
                    help='length of sequence used to condition sequence model for action prediction')

parser.add_argument('--hidden_dims', metavar='N', type=int, default = 256,
                    help='length of sequence used to condition sequence model for action prediction')

parser.add_argument('--batch_size', metavar='N', type=int, default = 128,
                    help='length of sequence used to condition sequence model for action prediction')
parser.add_argument('--chunk_length', metavar='N', type=int, default = 50,
                    help='length of sequence used to condition sequence model for action prediction')

parser.add_argument('--lmbd', metavar='N', type=float, default = 0.1,
                    help='alpha used to encourage compression and exploration simultaniously')
parser.add_argument('--alpha', metavar='N', type=float, default = 0.1,
                    help='alpha used to encourage compression and exploration simultaniously')

parser.add_argument('--rho', metavar='N', type=float, default = 0.01,
                    help='rho used for the critic soft update')

parser.add_argument('--lr', metavar='N', type=float, default = 0.001,
                    help='rho used for the critic soft update')

parser.add_argument('--dom_name', metavar='N', type=str, default = "cartpole",
                    help='suite domain name')
parser.add_argument('--task_name', metavar='N', type=str, default = "balance",
                    help='suite domain name')
parser.add_argument('--eval_freq', metavar='N', type=int, default = 20,
                    help='length of sequence used to condition sequence model for action prediction')


parser.add_argument('--quantization_res', metavar='N', type=int, default = 100,
                    help='length of sequence used to condition sequence model for action prediction')

parser.add_argument('--num_test_episodes', metavar='N', type=int, default = 20,
                    help='length of sequence used to condition sequence model for action prediction')

parser.add_argument('--model_name', metavar='N', type=str, default = "sac",
                    help='suite domain name')



def train_sac_dmc(seed, args, run):


    env = suite.load(domain_name=args.dom_name, task_name=args.task_name, task_kwargs={'random': 0})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = compute_num_features(env)
    action_spec = env.action_spec()
    action_dims = action_spec.shape[0]
    episode_length = int(1000/args.action_repeat)
    test_scores = torch.zeros(int(args.num_episodes/args.eval_freq)+1, args.num_test_episodes)
    eval_counter = 0

    num_random = args.action_repeat # we want 1000 random seed data points, since there are always 1000/action_repeat observations in an episode, we set num_random episodes to action _repeat

    if args.model_name == 'sac':
        agent = SACContinuous(state_dims=num_features, action_dims = action_dims, hidden_dims=args.hidden_dims, gamma=0.99, alpha=args.alpha, rho=args.rho).to(device=device)
    else:
        raise ValueError('model name not recognized')

    optimizer_actor = optim.Adam(list(agent.policy_head.parameters()),lr=args.lr)
    optimizer_critics = optim.Adam(list(agent.qsrc1.parameters()) + list(agent.qsrc2.parameters()), lr=args.lr)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = suite.load(domain_name=args.dom_name, task_name=args.task_name, task_kwargs={'random': seed})
    buffer = Buffer(episode_length=episode_length, buffer_size=1000000, batch_size = args.batch_size)

    for i in range(args.num_episodes):
        time_step = env.reset()
        current_state = get_dmc_state(env, time_step) #torch.cat(tuple(torch.tensor(val).view(-1, 1) for val in time_step.observation.values())).T.squeeze(0)
        step = 0
        S = torch.zeros(episode_length, num_features)
        A = torch.zeros(episode_length, action_dims)
        S_prime = torch.zeros(episode_length, num_features)
        R = torch.zeros(episode_length, 1)
        terminal = torch.zeros(episode_length, 1)
        while not time_step.last():

            S[step] = current_state
            if i < num_random:
                action = np.random.uniform(-1, 1, (1, action_dims))
            else:
                action = agent.act(current_state.float().to(device)).detach().cpu().numpy()

            time_step, reward = env_step_repeat(env, action, n=args.action_repeat)
            R[step] = reward
            next_state =  get_dmc_state(env, time_step)
            A[step] = torch.from_numpy(action)
            S_prime[step] = next_state

            current_state = next_state


            if i >= 4:

                critic_loss = agent.train_critic(buffer)
                optimizer_critics.zero_grad()
                critic_loss.backward()
                optimizer_critics.step()
                if step %2 == 0:
                #    print('hello')
                    actor_loss, alpha_loss = agent.train_actor()
                    optimizer_actor.zero_grad()
                    actor_loss.backward()
                    optimizer_actor.step()

                    agent.soft_update()
                    run.log({"actor_loss": actor_loss.item(), "alpha_loss": alpha_loss.item()}, commit=False)
                run.log({"critic_loss": critic_loss.item(), })



            step += 1
        #print(i, 'rewards: ', R.sum().item(), end = '\r')

        buffer.append(S, A, S_prime, R, terminal)
        buffer.finish_episode()

        if i % args.eval_freq == 0:
            test_rewards = run_test_episodes(env, agent, repeats=args.action_repeat, num_episodes=args.num_test_episodes, pixels = False)
            test_scores[eval_counter] = test_rewards
            eval_counter+=1
            run.log({"test_rewards": test_rewards.mean().item()})
            print(i, 'test r ', test_rewards.mean().item(), flush=True)
    test_rewards = run_test_episodes(env, agent, repeats=args.action_repeat, num_episodes=args.num_test_episodes, pixels = False)
    test_scores[eval_counter] = test_rewards
    run.log({"test_rewards": test_rewards.mean().item()})
    print(i, 'test r ', test_rewards.mean().item(), flush=True)
    agent.compression_algo = None
    return agent, test_scores


if __name__ == "__main__":

    args, unknown = parser.parse_known_args()

    wandb.login()

    print(args, flush=True)
    agents = []
    all_scores = torch.zeros(args.num_seeds, int(args.num_episodes/args.eval_freq)+1, args.num_test_episodes)
    run = wandb.init(
        project="JSC",
        name="test_experiment",
    )

    for seed in range(1):
        agent, scores = train_sac_dmc(seed, args, run)
        agents.append(agent)
        all_scores[seed] = scores
