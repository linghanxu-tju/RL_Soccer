import os
import gym
import numpy as np
import torch
from copy import  deepcopy
from torch.utils.tensorboard import SummaryWriter
from OppModeling.atari_wrappers import make_ftg_ram, make_ftg_ram_nonstation
from OppModeling.SAC import MLPActorCritic

def test_proc(global_ac, env, args, device):
    scores, wins, m_score, win_rate = [], [], 0, 0
    o, ep_ret, ep_len = env.reset(), 0, 0
    discard = False
    local_t = 0
    local_e = 0
    while local_e < args.test_episode:
        with torch.no_grad():
            a = global_ac.get_action(o, greedy=True, device=device)
        # Step the env
        o2, r, d, info = env.step(a)
        if info.get('no_data_receive', False):
            discard = True
        ep_ret += r
        ep_len += 1
        d = False if (ep_len == args.max_ep_len) or discard else d
        o = o2
        local_t += 1
        # End of trajectory handling
        if d or (ep_len == args.max_ep_len) or discard:
            # logger.store(EpRet=ep_ret, EpLen=ep_len)
            local_e += 1
            if info.get('win', False):
                wins.append(1)
            else:
                wins.append(0)
            scores.append(ep_ret)
            o, ep_ret, ep_len = env.reset(), 0, 0
            discard = False
    m_score = np.mean(scores)
    win_rate = np.mean(wins)
    return m_score, win_rate, local_t


def test_summary(p2, steps, m_score, win_rate, writer, args, e):
    print("\n" + "=" * 20 + "TEST SUMMARY" + "=" * 20)
    summary = "opponent:\t{}\n# test episode:\t{}\n# total steps:\t{}\nmean score:\t{:.1f}\nwin_rate:\t{}".format(
        p2, args.test_episode, steps, m_score, win_rate)
    print(summary)
    print("=" * 20 + "TEST SUMMARY" + "=" * 20 + "\n")
    writer.add_scalar("Test/mean_score", m_score.item(), e)
    writer.add_scalar("Test/win_rate", win_rate.item(), e)
    writer.add_scalar("Test/total_step", steps, e)


def test_func(test_q, rank, E, p2, args, device, tensorboard_dir, ):
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    print("set up Test process env")
    temp_dir = os.path.join(tensorboard_dir, "test_{}".format(p2))
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    writer = SummaryWriter(log_dir=temp_dir)
    # non_station evaluation
    if args.exp_name == "test":
        env = gym.make("CartPole-v0")
    elif p2 == "Non-station":
        env = make_ftg_ram_nonstation(args.env, p2_list=args.list, total_episode=args.test_episode,
                                      stable=args.stable)
    else:
        env = make_ftg_ram(args.env, p2=p2)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    ac_kwargs = dict(hidden_sizes=[args.hid] * args.l)
    if args.cpc:
        local_ac = MLPActorCritic(obs_dim + args.c_dim, act_dim, **ac_kwargs)
    else:
        local_ac = MLPActorCritic(obs_dim, act_dim, **ac_kwargs)
    env.close()
    del env
    # Main loop: collect experience in env and update/log each epoch
    while E.value() <= args.episode:
        received_obj = test_q.get()
        e = E.value()
        print("TEST Process {} loaded new mode".format(rank))
        model_dict = deepcopy(received_obj)
        local_ac.load_state_dict(model_dict)
        del received_obj
        if args.exp_name == "test":
            env = gym.make("CartPole-v0")
        elif p2 == "Non-station":
            env = make_ftg_ram_nonstation(args.env, p2_list=args.list, total_episode=args.test_episode,stable=args.stable)
        else:
            env = make_ftg_ram(args.env, p2=p2)
        print("TESTING process {} start to test, opp: {}".format(rank, p2))
        m_score, win_rate, steps = test_proc(local_ac, env, args, device)
        test_summary(p2, steps, m_score, win_rate, writer, args, e)
        env.close()
        del env
        print("TESTING process {} finished, opp: {}".format(rank, p2))
