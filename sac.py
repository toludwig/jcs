import argparse
import wandb
from train import train_sac

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task_name', type=str, default='Juggler')
    parser.add_argument('--pattern', type=str, default='[3,0,0]')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--rho', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--hidden_dims', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    args.pattern = eval(args.pattern)

    print(args.pattern)
    wandb.login()

    run = wandb.init(
        entity="tobiludw-university-t-bingen",
        project="JSC",
        name="test_experiment",
    )
    train_sac(args.seed, run, args)











