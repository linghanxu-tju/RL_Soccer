import argparse
import json
import os
import gym
import time
import torch
import copy
import numpy as np
import torch.nn as nn
import torch.multiprocessing as mp
from torch.optim import Adam
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from OppModeling.ReplayBuffer import ReplayBuffer, ReplayBufferOppo
from OppModeling.utils import Counter, count_vars
from OppModeling.SAC import MLPActorCritic
from OppModeling.CPC import CPC
from OppModeling.train_apex import sac
from games import Soccer, SoccerPLUS
from OppModeling.model_parameter_trans import state_dict_trans

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
    # sac setting
    parser.add_argument('--replay_size', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2, help="layers")
    parser.add_argument('--episode', type=int, default=100000)
    parser.add_argument('--update_after', type=int, default=500)
    parser.add_argument('--update_every', type=int, default=1)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--min_alpha', type=float, default=0.05)
    parser.add_argument('--dynamic_alpha', default=False, action="store_true")
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--polyak', type=float, default=0.995)
    # CPC setting
    parser.add_argument('--cpc', default=False, action="store_true")
    parser.add_argument('--cpc_batch', type=int, default=128)
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--c_dim', type=int, default=5)
    parser.add_argument('--timestep', type=int, default=5)
    parser.add_argument('--cpc_update_freq', type=int, default=1,)
    parser.add_argument('--forget_percent', type=float, default=0.2,)

    # evaluation settings
    parser.add_argument('--test_episode', type=int, default=10)
    parser.add_argument('--test_every', type=int, default=100)
    # Saving settings
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='SoccerPlus')
    parser.add_argument('--save-dir', type=str, default="./experiments")
    parser.add_argument('--traj_dir', type=str, default="./experiments")
    parser.add_argument('--model_para', type=str, default="sac.torch")
    parser.add_argument('--cpc_para', type=str, default="cpc.torch")
    parser.add_argument('--numpy_para', type=str, default="model.numpy")
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
    main_dir = os.path.join(tensorboard_dir, "train")
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    writer = SummaryWriter(log_dir=main_dir)
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
    env = SoccerPLUS()
    obs_dim = env.n_features
    act_dim = env.n_actions
    # create model
    global_ac = MLPActorCritic(obs_dim, act_dim, **ac_kwargs)
    var_counts = tuple(count_vars(module) for module in [global_ac.pi, global_ac.q1, global_ac.q2])
    print('\nNumber of SAC parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

    if args.cpc:
        global_cpc = CPC(timestep=args.timestep, obs_dim=obs_dim, hidden_sizes=[args.hid] * args.l, z_dim=args.z_dim,
                         c_dim=args.c_dim, device=device)
        var_counts = tuple(count_vars(module) for module in [global_cpc])
        print('\nNumber of CPC parameters: \t%d, \n' % var_counts)
    else:
        global_cpc = None
    # create shared model for actor
    global_ac_targ = deepcopy(global_ac)
    shared_ac = deepcopy(global_ac).cpu()
    # create optimizer
    pi_optimizer = Adam(global_ac.pi.parameters(), lr=args.lr, eps=1e-4)
    q1_optimizer = Adam(global_ac.q1.parameters(), lr=args.lr, eps=1e-4)
    q2_optimizer = Adam(global_ac.q2.parameters(), lr=args.lr, eps=1e-4)
    alpha_optim = Adam([global_ac.log_alpha], lr=args.lr, eps=1e-4)
    if args.cpc:
        cpc_optimizer = Adam(global_cpc.parameters(), lr=args.lr, eps=1e-4)
    env.close()
    del env

    # training setup
    T = Counter()  # training steps
    E = Counter()  # training episode
    replay_buffer = ReplayBufferOppo(obs_dim=obs_dim, max_size=args.replay_size, cpc=args.cpc,
                                     cpc_model=global_cpc, writer=writer)

    if os.path.exists(os.path.join(args.save_dir, args.exp_name, args.model_para)):
        global_ac.load_state_dict(torch.load(os.path.join(args.save_dir, args.exp_name, args.model_para)))
        print("load sac model")
        if args.cpc:
            global_cpc.load_state_dict(torch.load(os.path.join(args.save_dir, args.exp_name, args.cpc_para)))
            print("load cpc model")
    if os.path.exists(os.path.join(args.save_dir, args.exp_name, args.train_indicator)):
        (e, t) = torch.load(os.path.join(args.save_dir, args.exp_name, args.train_indicator))
        T.set(t)
        E.set(e)
        print("load training indicator")

    last_updated = 0
    last_train = 0
    last_saved = 0
    if args.cuda:
        global_ac.to(device)
        global_ac_targ.to(device)
        if args.cpc:
            global_cpc.to(device)

    for p in global_ac_targ.parameters():
        p.requires_grad = False

    buffer_q = mp.SimpleQueue()
    model_q = [mp.SimpleQueue() for _ in range(args.n_process)]
    processes = []
    # Process 0 for evaluation
    for rank in range(args.n_process):  # 4 test process
        model_q[rank].put(shared_ac.state_dict())
        # Test during training
        # if rank == 0:
        #     p = mp.Process(target=test_func, args=(test_q, rank, E, "Non-station", args, torch.device("cpu"),tensorboard_dir))
        # elif rank < 4:
        #     p = mp.Process(target=test_func, args=(test_q, rank, E, args.list[(rank-1) % len(args.list)], args, torch.device("cpu"), tensorboard_dir))
        # else:
        #     p = mp.Process(target=sac, args=(model_q, rank, E, args,  buffer_q, torch.device("cpu"), tensorboard_dir))
        p = mp.Process(target=sac, args=(rank, E, args, model_q[rank], buffer_q, torch.device("cpu"), tensorboard_dir))
        p.start()
        processes.append(p)

    target_entropy = -np.log((1.0 / act_dim)) * 0.5
    alpha = max(global_ac.log_alpha.exp().item(), args.min_alpha) if args.dynamic_alpha else args.min_alpha
    # alpha = args.min_alpha
    while E.value() <= args.episode:
        # receive data from actors, will block if no data received
        # print("Going to read data from ACTOR......")
        received_data = buffer_q.get()
        # print("Finish Reading data from ACTOR!!!!!!")
        (trajectory, meta) = copy.deepcopy(received_data)
        del received_data
        replay_buffer.store(trajectory, meta=meta)
        E.increment()
        t = T.value()
        e = E.value()

        # SAC Update handling
        if e >= args.update_after and e % args.update_every == 0 and e != last_train:
            # if the batch size is very large, can train sac once per round
            for _ in range(args.update_every):
                batch = replay_buffer.sample_trans(args.batch_size, device=device)
                # First run one gradient descent step for Q1 and Q2
                q1_optimizer.zero_grad()
                q2_optimizer.zero_grad()

                loss_q = global_ac.compute_loss_q(batch, global_ac_targ, args.gamma, alpha)
                loss_q.backward()
                nn.utils.clip_grad_norm_(global_ac.parameters(), max_norm=10, norm_type=2)
                q1_optimizer.step()
                q2_optimizer.step()

                # Next run one gradient descent step for pi.
                pi_optimizer.zero_grad()
                loss_pi, entropy = global_ac.compute_loss_pi(batch, alpha)
                loss_pi.backward()
                nn.utils.clip_grad_norm_(global_ac.parameters(), max_norm=10, norm_type=2)
                pi_optimizer.step()

                alpha_optim.zero_grad()
                alpha_loss = -(global_ac.log_alpha * (entropy + target_entropy).detach()).mean()
                alpha_loss.backward(retain_graph=False)
                nn.utils.clip_grad_norm_(global_ac.parameters(), max_norm=10, norm_type=2)
                alpha = max(global_ac.log_alpha.exp().item(), args.min_alpha) if args.dynamic_alpha else args.min_alpha
                alpha_optim.step()

                # Finally, update target networks by polyak averaging.
                with torch.no_grad():
                    for p, p_targ in zip(global_ac.parameters(), global_ac_targ.parameters()):
                        p_targ.data.copy_((1 - args.polyak) * p.data + args.polyak * p_targ.data)

                writer.add_scalar("learner/pi_loss", loss_pi.detach().item(), t)
                writer.add_scalar("learner/q_loss", loss_q.detach().item(), t)
                writer.add_scalar("learner/alpha_loss", alpha_loss.detach().item(), t)
                writer.add_scalar("learner/alpha", alpha, t)
                writer.add_scalar("learner/entropy", entropy.detach().mean().item(), t)
                writer.add_scalar("learner/buffer_size", replay_buffer.size, t)
                last_train = e
                T.increment()

        # CPC update handing
        if args.cpc and e > args.cpc_batch and e % args.cpc_update_freq == 0:
            for _ in range(args.cpc_update_freq * 10):
                data, indexes, min_len = replay_buffer.sample_traj(args.cpc_batch)
                cpc_optimizer.zero_grad()
                c_hidden = global_cpc.init_hidden(len(data), args.c_dim)
                acc, loss, latents = global_cpc(data, c_hidden)

                # replay_buffer.update_latent(indexes, min_len, latents.detach())
                loss.backward()
                # add gradient clipping
                nn.utils.clip_grad_norm_(global_cpc.parameters(), max_norm=20, norm_type=2)
                cpc_optimizer.step()
                writer.add_scalar("learner/cpc_acc", acc, e)
                writer.add_scalar("learner/cpc_loss", loss.detach().item(), e)
                c_hidden = global_cpc.init_hidden(1, args.c_dim)

        # CPC latent update
        if args.cpc and e > args.cpc_batch and e % 500 == 0:
            replay_buffer.create_latents(e=e)

        # deliver the model and save
        if e % args.save_freq == 0 and e > 0 and e != last_saved:
            temp = copy.deepcopy(global_ac).cpu()
            shared_ac_state_dict = copy.deepcopy(temp.state_dict())
            for i in range(args.n_process):
                model_q[i].put(shared_ac_state_dict, )
            torch.save(global_ac.state_dict(), os.path.join(experiment_dir, args.model_para))
            if args.cpc:
                torch.save(global_cpc.state_dict(), os.path.join(experiment_dir, args.cpc_para))
            # state_dict_trans(global_ac.state_dict(), os.path.join(experiment_dir, args.numpy_para))
            # torch.save((e, t, list(scores), list(wins)), os.path.join(args.save_dir, args.exp_name, "model_data_{}".format(e)))
            print("Saving model at episode:{}".format(t))
            last_saved = e

        # if e > 0 and e % args.test_every == 0 and tested_e != e:
        #     temp = copy.deepcopy(global_ac).cpu()
        #     shared_ac_state_dict = copy.deepcopy(temp.state_dict())
        #     for _ in range(4):
        #         test_q.put(shared_ac_state_dict,)
        #     tested_e = e
