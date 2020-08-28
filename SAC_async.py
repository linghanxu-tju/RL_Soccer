import argparse
import json
import os
import gym
import time
import torch
import torch.multiprocessing as mp
from copy import deepcopy
from games import Soccer
# from tensorboardX import GlobalSummaryWriter
from OppModeling.atari_wrappers import make_ftg_ram, make_ftg_ram_nonstation
from OppModeling.utils import Counter, count_vars
from OppModeling.SAC import MLPActorCritic
from OppModeling.CPC import CPC
from OppModeling.train_sac import sac, sac_opp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # running setting
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--n_process', type=int, default=4)
    # basic env setting
    parser.add_argument('--env', type=str, default="FightingiceDataFrameskip-v0")
    parser.add_argument('--p2', type=str, default="Toothless")
    # non station agent settings
    parser.add_argument('--non_station', default=False, action='store_true')
    parser.add_argument('--stable', default=False, action='store_true')
    parser.add_argument('--station_rounds', type=int, default=1000)
    parser.add_argument('--list', nargs='+')
    # training setting
    parser.add_argument('--replay_size', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=4, help="layers")
    parser.add_argument('--episode', type=int, default=100000)
    parser.add_argument('--start_steps', type=int, default=1000)
    parser.add_argument('--update_after', type=int, default=100)
    parser.add_argument('--update_every', type=int, default=1)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--min_alpha', type=float, default=0.05)
    parser.add_argument('--fix_alpha', default=False, action="store_true")
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--polyak', type=float, default=0.995)
    # CPC setting
    parser.add_argument('--cpc', default=False, action="store_true")
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--c_dim', type=int, default=32)
    parser.add_argument('--timestep', type=int, default=10)
    # OOD setting
    parser.add_argument('--ood', default=False, action="store_true")
    parser.add_argument('--ood_K', type=int, default=13)
    parser.add_argument('--ood_starts', type=int, default=100)
    parser.add_argument('--ood_train_per', type=float, default=0.25)
    parser.add_argument('--ood_update_rounds', type=int, default=50)
    parser.add_argument('--ood_drop_lower',type=int, default=10)
    parser.add_argument('--ood_drop_upper', type=int, default=60)
    # Saving settings
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--save-dir', type=str, default="./experiments")
    parser.add_argument('--traj_dir', type=str, default="./experiments")
    parser.add_argument('--model_para', type=str, default="test_sac.torch")
    parser.add_argument('--cpc_para', type=str, default="test_cpc.torch")
    parser.add_argument('--numpy_para', type=str, default="test.numpy")
    parser.add_argument('--train_indicator', type=str, default="test.data")
    args = parser.parse_args()

    # Basic Settings
    mp.set_start_method("forkserver")
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(torch.get_num_threads())
    experiment_dir = os.path.join(args.save_dir, args.exp_name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    tensorboard_dir = os.path.join(experiment_dir, "runs")
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    with open(os.path.join(experiment_dir, "arguments"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    device = torch.device("cuda") if args.cuda else torch.device("cpu")
    # env and model setup
    ac_kwargs = dict(hidden_sizes=[args.hid] * args.l)
    # if args.exp_name == "test":
    #     env = gym.make("CartPole-v0")
    # elif args.non_station:
    #     env = make_ftg_ram_nonstation(args.env, p2_list=args.list, total_episode=args.station_rounds,stable=args.stable)
    # else:
    #     env = make_ftg_ram(args.env, p2=args.p2)
    # obs_dim = env.observation_space.shape[0]
    # act_dim = env.action_space.n
    env = Soccer()
    # env = gym.make("CartPole-v0")
    obs_dim = env.n_features
    act_dim = env.n_actions
    if args.cpc:
        global_ac = MLPActorCritic(obs_dim+args.c_dim, act_dim, **ac_kwargs)
        global_cpc = CPC(timestep=args.timestep, obs_dim=obs_dim, hidden_sizes=[args.hid] * args.l, z_dim=args.z_dim,c_dim=args.c_dim)
        global_cpc.share_memory()
    else:
        global_ac = MLPActorCritic(obs_dim, act_dim, **ac_kwargs)
        global_cpc = None

    # async training setup
    T = Counter()
    E = Counter()
    scores = mp.Manager().list()
    wins = mp.Manager().list()
    buffer = mp.Manager().list()

    if os.path.exists(os.path.join(args.save_dir, args.exp_name, args.model_para)):
        global_ac.load_state_dict(torch.load(os.path.join(args.save_dir, args.exp_name, args.model_para)))
        print("load sac model")
        if args.cpc:
            global_cpc.load_state_dict(torch.load(os.path.join(args.save_dir, args.exp_name, args.cpc_para)))
            print("load cpc model")
    if os.path.exists(os.path.join(args.save_dir, args.exp_name, args.train_indicator)):
        (e, t, scores_list, wins_list) = torch.load(os.path.join(args.save_dir, args.exp_name, args.train_indicator))
        T.set(t)
        E.set(e)
        scores.extend(scores_list)
        wins.extend(wins_list)
        print("load training indicator")

    global_ac_targ = deepcopy(global_ac)
    env.close()
    del env
    if args.cuda:
        global_ac.to(device)
        global_ac_targ.to(device)
        if args.cpc:
            global_cpc.to(device)
    global_ac.share_memory()
    global_ac_targ.share_memory()
    var_counts = tuple(count_vars(module) for module in [global_ac.pi, global_ac.q1, global_ac.q2])
    print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

    processes = []
    for rank in range(args.n_process):
        if args.cpc:
            train_func = sac_opp
            n_args = (global_ac, global_ac_targ, global_cpc, rank, T, E, args, scores, wins, buffer)
        else:
            train_func = sac
            n_args = (global_ac, global_ac_targ, rank, T, E, args, scores, wins, buffer)
        kwargs = dict(device=device, tensorboard_dir=tensorboard_dir)

        p = mp.Process(target=train_func, args=n_args, kwargs=kwargs)
        p.start()
        # time.sleep(5)
        processes.append(p)
    for p in processes:
        p.join()


