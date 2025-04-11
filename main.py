import  argparse
import  wandb
from sympy.printing.tree import print_node
from train import train_sac

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task_name', type=str, default='Juggler')
    parser.add_argument('--pattern', type=str, default='[3,3,3]')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--rho', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--hidden_dims', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    args.pattern = eval(args.pattern)

    print(args.pattern)
    wandb.login(key='47ce5a0f1ac4744586c8eb6cad968c2f03bf197c')

    run = wandb.init(
        project="JSC",
        name="test_experiment",
    )
    train_sac(args.seed, run, args)









