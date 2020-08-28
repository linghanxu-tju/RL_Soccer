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
from games import Soccer
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from OppModeling.ReplayBuffer import ReplayBuffer, ReplayBufferOppo, ReplayBufferShare
from OppModeling.atari_wrappers import make_ftg_ram, make_ftg_ram_nonstation
from OppModeling.utils import Counter, count_vars
from OppModeling.SAC import MLPActorCritic
from OppModeling.CPC import CPC
from OppModeling.train_apex import sac
from games import Soccer
from OppModeling.evaluation_apex import test_func
from OppModeling.model_parameter_trans import state_dict_trans
from OOD.glod import convert_to_glod, retrieve_scores,ood_scores

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
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2, help="layers")
    parser.add_argument('--episode', type=int, default=100000)
    parser.add_argument('--update_after', type=int, default=100)
    parser.add_argument('--update_every', type=int, default=1)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--min_alpha', type=float, default=0.3)
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
    # evaluation settings
    parser.add_argument('--test_episode', type=int, default=10)
    parser.add_argument('--test_every', type=int, default=100)
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
    main_dir = os.path.join(tensorboard_dir, "Main")
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
    env = Soccer()
    obs_dim = env.n_features
    act_dim = env.n_actions
    if args.cpc:
        global_ac = MLPActorCritic(obs_dim+args.c_dim, act_dim, **ac_kwargs)
        global_cpc = CPC(timestep=args.timestep, obs_dim=obs_dim, hidden_sizes=[args.hid] * args.l, z_dim=args.z_dim,c_dim=args.c_dim)
    else:
        global_ac = MLPActorCritic(obs_dim, act_dim, **ac_kwargs)
        global_cpc = None
    # create shared model for actor
    shared_ac = deepcopy(global_ac).cpu()
    global_ac_targ = deepcopy(global_ac)

    target_entropy = -np.log((1.0 / act_dim)) * 0.5
    alpha = max(global_ac.log_alpha.exp().item(), args.min_alpha) if not args.fix_alpha else args.min_alpha
    pi_optimizer = Adam(global_ac.pi.parameters(), lr=args.lr, eps=1e-4)
    q1_optimizer = Adam(global_ac.q1.parameters(), lr=args.lr, eps=1e-4)
    q2_optimizer = Adam(global_ac.q2.parameters(), lr=args.lr, eps=1e-4)
    alpha_optim = Adam([global_ac.log_alpha], lr=args.lr, eps=1e-4)
    env.close()
    del env

    # training setup
    T = Counter()
    E = Counter()
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, size=args.replay_size)
    training_buffer = ReplayBuffer(obs_dim=obs_dim, size=args.replay_size)

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

    glod_model = None
    glod_lower = None
    glod_upper = None
    last_updated = 0
    last_train = 0
    last_saved = 0
    tested_e = 0
    if args.cuda:
        global_ac.to(device)
        global_ac_targ.to(device)
        if args.cpc:
            global_cpc.to(device)

    for p in global_ac_targ.parameters():
        p.requires_grad = False

    var_counts = tuple(count_vars(module) for module in [global_ac.pi, global_ac.q1, global_ac.q2])
    print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

    buffer_q = mp.SimpleQueue()
    model_q = mp.SimpleQueue()
    test_q = mp.SimpleQueue()
    processes = []
    # Process 0 for evaluation
    for rank in range(args.n_process): # 4 test process
        model_q.put(shared_ac.state_dict())
        # Test during training
        # if rank == 0:
        #     p = mp.Process(target=test_func, args=(test_q, rank, E, "Non-station", args, torch.device("cpu"),tensorboard_dir))
        # elif rank < 4:
        #     p = mp.Process(target=test_func, args=(test_q, rank, E, args.list[(rank-1) % len(args.list)], args, torch.device("cpu"), tensorboard_dir))
        # else:
        #     p = mp.Process(target=sac, args=(model_q, rank, E, args,  buffer_q, torch.device("cpu"), tensorboard_dir))
        p = mp.Process(target=sac, args=(model_q, rank, E, args, buffer_q, torch.device("cpu"), tensorboard_dir))
        p.start()
        if not args.exp_name == "test":
            time.sleep(10)
        processes.append(p)

    while E.value() <= args.episode:
        # receive data from actors
        # print("Going to read data from ACTOR......")
        received_obj = buffer_q.get()
        # print("Finish Reading data from ACTOR!!!!!!")
        transition = copy.deepcopy(received_obj)
        del received_obj

        (o, a, r, o2, d, p2) = transition
        if glod_model is None or not args.ood:
            replay_buffer.store(o, a, r, o2, d, p2)
            training_buffer.store(o, a, r, o2, d, p2)
        else:
            obs_glod_score = retrieve_scores(glod_model, np.expand_dims(o, axis=0), device=torch.device("cpu"), k=args.ood_K)
            if glod_lower <= obs_glod_score <= glod_upper:
                training_buffer.store(o, a, r, o2, d, p2)
            replay_buffer.store(o, a, r, o2, d, p2)

        T.increment()
        t = T.value()
        e = E.value()

        # OOD update stage, can only use CPU as the GPU memory can not hold so much data
        if e >= args.ood_starts and e % args.ood_update_rounds == 0 and args.ood and e != last_updated:
            print("OOD updating at rounds {}".format(e))
            print("Replay Buffer Size: {}, Training Buffer Size: {}".format(replay_buffer.size, training_buffer.size))
            glod_idxs = np.random.randint(0, training_buffer.size, size=int(training_buffer.size * args.ood_train_per))
            glod_input = training_buffer.obs_buf[glod_idxs]
            glod_target = training_buffer.act_buf[glod_idxs]
            ood_train = (glod_input, glod_target)
            glod_model = deepcopy(global_ac.pi).cpu()
            glod_model = convert_to_glod(glod_model, train_loader=ood_train, hidden_dim=args.hid, act_dim=act_dim, device=torch.device("cpu"))
            training_buffer = deepcopy(replay_buffer)
            glod_scores = retrieve_scores(glod_model, replay_buffer.obs_buf[:training_buffer.size], device=torch.device("cpu"),
                                          k=args.ood_K)
            glod_scores = glod_scores.detach().cpu().numpy()
            glod_p2 = training_buffer.p2_buf[:training_buffer.size]
            drop_points = np.percentile(a=glod_scores, q=[args.ood_drop_lower, args.ood_drop_upper])
            glod_lower, glod_upper = drop_points[0], drop_points[1]
            mask = np.logical_and((glod_scores >= glod_lower), (glod_scores <= glod_upper))
            reserved_indexes = np.argwhere(mask).flatten()
            if len(reserved_indexes) > 0:
                training_buffer.ood_drop(reserved_indexes)
            writer.add_histogram(values=glod_scores, max_bins=300, global_step=e, tag="OOD")
            print("Replay Buffer Size: {}, Training Buffer Size: {}".format(replay_buffer.size, training_buffer.size))
            torch.save((glod_scores, replay_buffer.p2_buf[:replay_buffer.size]),
                       os.path.join(args.save_dir, args.exp_name, "glod_info_{}".format(e)))
            last_updated = e

        # SAC Update handling
        if e >= args.update_after and t % args.update_every == 0 and t != last_train:
            batch = training_buffer.sample_trans(args.batch_size, device=device)
            # First run one gradient descent step for Q1 and Q2
            q1_optimizer.zero_grad()
            q2_optimizer.zero_grad()
            pi_optimizer.zero_grad()
            alpha_optim.zero_grad()

            loss_q = global_ac.compute_loss_q(batch, global_ac_targ, args.gamma, alpha)
            loss_q.backward()

            # Next run one gradient descent step for pi.
            loss_pi, entropy = global_ac.compute_loss_pi(batch, alpha)
            loss_pi.backward()

            alpha_loss = -(global_ac.log_alpha * (entropy + target_entropy).detach()).mean()
            alpha_loss.backward(retain_graph=False)
            alpha = max(global_ac.log_alpha.exp().item(), args.min_alpha) if not args.fix_alpha else args.min_alpha

            nn.utils.clip_grad_norm_(global_ac.parameters(), max_norm=20, norm_type=2)
            pi_optimizer.step()
            q1_optimizer.step()
            q2_optimizer.step()
            alpha_optim.step()

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(global_ac.parameters(), global_ac_targ.parameters()):
                    p_targ.data.copy_((1 - args.polyak) * p.data + args.polyak * p_targ.data)

            writer.add_scalar("training/pi_loss", loss_pi.detach().item(), t)
            writer.add_scalar("training/q_loss", loss_q.detach().item(), t)
            writer.add_scalar("training/alpha_loss", alpha_loss.detach().item(), t)
            writer.add_scalar("training/entropy", entropy.detach().mean().item(), t)
            last_train = t

        # deliver the model and save
        if e % args.save_freq == 0 and e > 0 and e != last_saved:
            temp = copy.deepcopy(global_ac).cpu()
            shared_ac_state_dict = copy.deepcopy(temp.state_dict())
            for _ in range(args.n_process):
                model_q.put(shared_ac_state_dict,)
            torch.save(global_ac.state_dict(), os.path.join(args.save_dir, args.exp_name, "model_torch_{}".format(e)))
            state_dict_trans(global_ac.state_dict(), os.path.join(args.save_dir, args.exp_name,  "model_numpy_{}".format(e)))
            # torch.save((e, t, list(scores), list(wins)), os.path.join(args.save_dir, args.exp_name, "model_data_{}".format(e)))
            print("Saving model at episode:{}".format(t))
            last_saved = e

        # if e > 0 and e % args.test_every == 0 and tested_e != e:
        #     temp = copy.deepcopy(global_ac).cpu()
        #     shared_ac_state_dict = copy.deepcopy(temp.state_dict())
        #     for _ in range(4):
        #         test_q.put(shared_ac_state_dict,)
        #     tested_e = e
