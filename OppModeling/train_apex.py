import os
import gym
import time
import numpy as np
import torch
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from games import Soccer,SoccerPLUS
from Policy_New import Policy
from OppModeling.atari_wrappers import make_ftg_ram,make_ftg_ram_nonstation
from OppModeling.SAC import MLPActorCritic


def sac(rank, E, args, model_q, buffer_q, device=None, tensorboard_dir=None,):
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    # writer = GlobalSummaryWriter.getSummaryWriter()
    tensorboard_dir = os.path.join(tensorboard_dir, str(rank))
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    # if args.exp_name == "test":
    #     env = gym.make("CartPole-v0")
    # elif args.non_station:
    #     env = make_ftg_ram_nonstation(args.env, p2_list=args.list, total_episode=args.station_rounds,stable=args.stable)
    # else:
    #     env = make_ftg_ram(args.env, p2=args.p2)
    # obs_dim = env.observation_space.shape[0]
    # act_dim = env.action_space.n
    env = SoccerPLUS(visual=True)
    opp_policy = Policy(game=env, player_num=False)
    obs_dim = env.n_features
    act_dim = env.n_actions
    ac_kwargs = dict(hidden_sizes=[args.hid] * args.l)
    local_ac = MLPActorCritic(obs_dim, act_dim, **ac_kwargs)
    print("set up child process env")

    # Prepare for interaction with environment
    scores, wins = [], []
    # meta data is purely for experiment analysis
    trajectory, meta = [], []
    o, ep_ret, ep_len = env.reset(), 0, 0
    discard = False
    local_t, local_e = 0, 0
    if not model_q.empty():
        print("Process {}\t Initially LOADING...".format(rank))
        received_obj = model_q.get()
        model_dict = deepcopy(received_obj)
        local_ac.load_state_dict(model_dict)
        print("Process {}\t Initially Loading FINISHED!!!".format(rank))
        del received_obj
    # Main loop: collect experience in env and update/log each epoch
    while E.value() <= args.episode:
        with torch.no_grad():
            if E.value() <= args.update_after:
                a = np.random.randint(act_dim)
            else:
                a = local_ac.get_action(o, device=device)

        # print(o)
        # Step the env
        o2, r, d, info = env.step(a, opp_policy.get_actions(1))
        env.render()
        if info.get('no_data_receive', False):
            discard = True
        ep_ret += r
        ep_len += 1

        d = False if (ep_len == args.max_ep_len) or discard else d
        # send the transition to main process
        if hasattr(env, 'p2'):
            opp = env.p2
        else:
            opp = None
        transition = (o, a, r, o2, d)
        trajectory.append(transition)
        meta.append([opp, rank, local_e, ep_len, r, a])
        o = o2
        local_t += 1
        # End of trajectory handling
        if d or (ep_len == args.max_ep_len) or discard:
            e = E.value()
            send_data = (trajectory, meta)
            buffer_q.put(send_data,)
            local_e += 1
            # logger.store(EpRet=ep_ret, EpLen=ep_len)
            if info.get('win', False):
                wins.append(1)
            else:
                wins.append(0)
            scores.append(ep_ret)
            m_score = np.mean(scores[-100:])
            win_rate = np.mean(wins[-100:])
            print(
                "Process\t{}\topponent:{},\t# of local episode :{},\tglobal episode {}\tround score: {},\tmean score : {:.1f},\twin rate:{},\tsteps: {}".format(
                    rank, args.p2, local_e, e, ep_ret, m_score, win_rate, ep_len))
            writer.add_scalar("actor/round_score", ep_ret, local_e)
            writer.add_scalar("actor/mean_score", m_score.item(), local_e)
            writer.add_scalar("actor/win_rate", win_rate.item(), local_e)
            writer.add_scalar("actor/round_step", ep_len, local_e)
            writer.add_scalar("actor/learner_actor_speed", e, local_e)
            o, ep_ret, ep_len = env.reset(), 0, 0
            discard = False
            trajectory, meta = list(), list()
            if not model_q.empty():
                print("Process {}\tLOADING model at Global\t{},local\t{} EPISODE...".format(rank, e, local_e))
                received_obj = model_q.get()
                model_dict = deepcopy(received_obj)
                local_ac.load_state_dict(model_dict)
                print("Process {}\tLOADED new mode at Global\t{},local\t{}!!!".format(rank, e, local_e))
                del received_obj

