import os
import gym
import numpy as np
import torch
from games import Soccer
import torch.nn as nn
from copy import deepcopy
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from OppModeling.SAC import MLPActorCritic
from OppModeling.ReplayBuffer import ReplayBuffer, ReplayBufferOppo, ReplayBufferShare
from OppModeling.atari_wrappers import make_ftg_ram,make_ftg_ram_nonstation
from OppModeling.model_parameter_trans import state_dict_trans
from OOD.glod import convert_to_glod, retrieve_scores,ood_scores


def sac(global_ac, global_ac_targ, rank, T, E, args, scores, wins, buffer_q, device=None, tensorboard_dir=None,):
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
    env = Soccer()
    obs_dim = env.n_features
    act_dim = env.n_actions
    print("set up child process env")
    local_ac = MLPActorCritic(obs_dim, act_dim, **dict(hidden_sizes=[args.hid] * args.l)).to(device)
    state_dict = global_ac.state_dict()
    local_ac.load_state_dict(state_dict)
    print("local ac load global ac")

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    # Async Version
    for p in global_ac_targ.parameters():
        p.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, size=args.replay_size)
    training_buffer = ReplayBuffer(obs_dim=obs_dim, size=args.replay_size)

    # Entropy Tuning
    target_entropy = -torch.prod(torch.Tensor(env.n_actions).to(device)).item()  # heuristic value from the paper
    alpha = max(local_ac.log_alpha.exp().item(), args.min_alpha) if not args.fix_alpha else args.min_alpha

    # Set up optimizers for policy and q-function
    # Async Version
    pi_optimizer = Adam(global_ac.pi.parameters(), lr=args.lr, eps=1e-4)
    q1_optimizer = Adam(global_ac.q1.parameters(), lr=args.lr, eps=1e-4)
    q2_optimizer = Adam(global_ac.q2.parameters(), lr=args.lr, eps=1e-4)
    alpha_optim = Adam([global_ac.log_alpha], lr=args.lr, eps=1e-4)

    # Prepare for interaction with environment
    o, ep_ret, ep_len = env.reset(), 0, 0
    discard = False
    glod_model = None
    glod_lower = None
    glod_upper = None
    last_updated = 0
    saved_e = 0
    t = T.value()
    e = E.value()
    local_t, local_e = 0, 0
    # Main loop: collect experience in env and update/log each epoch
    while e <= args.episode:

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.

        a = local_ac.get_action(o, device=device)
        if hasattr(env, 'p2'):
            p2 = env.p2
        else:
            p2 = None
        # Step the env
        o2, r, d, info = env.step(a,np.random.randint(act_dim))
        env.render()
        if glod_model is None or not args.ood:
            replay_buffer.store(o, a, r, o2, d, str(p2))
            training_buffer.store(o, a, r, o2, d, str(p2))
        else:
            obs_glod_score = retrieve_scores(glod_model, np.expand_dims(o, axis=0), device=torch.device("cpu"), k=args.ood_K)
            if glod_lower <= obs_glod_score <= glod_upper:
                training_buffer.store(o, a, r, o2, d, str(p2))
            replay_buffer.store(o, a, r, o2, d, str(p2))

        if info.get('no_data_receive', False):
            discard = True
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if (ep_len == args.max_ep_len) or discard else d
        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        T.increment()
        t = T.value()
        local_t += 1

        # End of trajectory handling
        if d or (ep_len == args.max_ep_len) or discard:
            E.increment()
            e = E.value()
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
                "Process {}, opponent:{}, # of global_episode :{},  # of global_steps :{}, round score: {}, mean score : {:.1f}, win_rate:{}, steps: {}, alpha: {}".format(
                    rank, args.p2, e, t, ep_ret, m_score, win_rate, ep_len, alpha))
            writer.add_scalar("metrics/round_score", ep_ret, e)
            writer.add_scalar("metrics/mean_score", m_score.item(), e)
            writer.add_scalar("metrics/win_rate", win_rate.item(), e)
            writer.add_scalar("metrics/round_step", ep_len, e)
            writer.add_scalar("metrics/alpha", alpha, e)
            o, ep_ret, ep_len = env.reset(), 0, 0
            discard = False

        # OOD update stage, can only use CPU as the GPU memory can not hold so much data
        if local_e >= args.ood_starts and local_e % args.ood_update_rounds == 0 and args.ood and local_e != last_updated:
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
            writer.add_histogram(values=glod_scores, max_bins=300, global_step=local_e, tag="OOD")
            print("Replay Buffer Size: {}, Training Buffer Size: {}".format(replay_buffer.size, training_buffer.size))
            torch.save((glod_scores, replay_buffer.p2_buf[:replay_buffer.size]),
                       os.path.join(args.save_dir, args.exp_name, "glod_info_{}".format(e)))
            last_updated = e

        # SAC Update handling
        if local_e >= args.update_after and local_t % args.update_every == 0:
            for j in range(args.update_every):

                batch = training_buffer.sample_trans(args.batch_size,device=device)
                # First run one gradient descent step for Q1 and Q2
                q1_optimizer.zero_grad()
                q2_optimizer.zero_grad()
                loss_q = local_ac.compute_loss_q(batch, global_ac_targ, args.gamma, alpha)
                loss_q.backward()

                # Next run one gradient descent step for pi.
                pi_optimizer.zero_grad()
                loss_pi, entropy = local_ac.compute_loss_pi(batch, alpha)
                loss_pi.backward()

                alpha_optim.zero_grad()
                alpha_loss = -(local_ac.log_alpha * (entropy + target_entropy).detach()).mean()
                alpha_loss.backward(retain_graph=False)
                alpha = max(local_ac.log_alpha.exp().item(), args.min_alpha) if not args.fix_alpha else args.min_alpha

                nn.utils.clip_grad_norm_(local_ac.parameters(), 20)
                for global_param, local_param in zip(global_ac.parameters(), local_ac.parameters()):
                    global_param._grad = local_param.grad

                pi_optimizer.step()
                q1_optimizer.step()
                q2_optimizer.step()
                alpha_optim.step()

                state_dict = global_ac.state_dict()
                local_ac.load_state_dict(state_dict)

                # Finally, update target networks by polyak averaging.
                with torch.no_grad():
                    for p, p_targ in zip(global_ac.parameters(), global_ac_targ.parameters()):
                        p_targ.data.copy_((1 - args.polyak) * p.data + args.polyak * p_targ.data)

                writer.add_scalar("training/pi_loss", loss_pi.detach().item(), t)
                writer.add_scalar("training/q_loss", loss_q.detach().item(), t)
                writer.add_scalar("training/alpha_loss", alpha_loss.detach().item(), t)
                writer.add_scalar("training/entropy", entropy.detach().mean().item(), t)

        if e % args.save_freq == 0 and e > 0 and e != saved_e:
            torch.save(global_ac.state_dict(), os.path.join(args.save_dir, args.exp_name, "model_torch_{}".format(e)))
            state_dict_trans(global_ac.state_dict(), os.path.join(args.save_dir, args.exp_name,  "model_numpy_{}".format(e)))
            torch.save((e, t, list(scores), list(wins)), os.path.join(args.save_dir, args.exp_name, "model_data_{}".format(e)))
            print("Saving model at episode:{}".format(t))
            saved_e = e


def sac_opp(global_ac, global_ac_targ, global_cpc, rank, T, E, args, scores, wins, buffer, device=None, tensorboard_dir=None, ):
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    # writer = GlobalSummaryWriter.getSummaryWriter()
    tensorboard_dir = os.path.join(tensorboard_dir, str(rank))
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    if args.exp_name == "test":
        env = gym.make("CartPole-v0")
    elif args.non_station:
        env = make_ftg_ram_nonstation(args.env, p2_list=args.list, total_episode=args.station_rounds,stable=args.stable)
    else:
        env = make_ftg_ram(args.env, p2=args.p2)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    print("set up child process env")
    local_ac = MLPActorCritic(obs_dim+args.c_dim, act_dim, **dict(hidden_sizes=[args.hid] * args.l)).to(device)
    local_ac.load_state_dict(global_ac.state_dict())
    print("local ac load global ac")

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    # Async Version
    for p in global_ac_targ.parameters():
        p.requires_grad = False

    # Experience buffer
    if args.cpc:
        replay_buffer = ReplayBufferOppo(obs_dim=obs_dim,max_size=args.replay_size,encoder=global_cpc)
    else:
        replay_buffer = ReplayBuffer(obs_dim=obs_dim, size=args.replay_size)

    # Entropy Tuning
    target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()  # heuristic value from the paper
    alpha = max(local_ac.log_alpha.exp().item(), args.min_alpha) if not args.fix_alpha else args.min_alpha

    # Set up optimizers for policy and q-function
    # Async Version
    pi_optimizer = Adam(global_ac.pi.parameters(), lr=args.lr, eps=1e-4)
    q1_optimizer = Adam(global_ac.q1.parameters(), lr=args.lr, eps=1e-4)
    q2_optimizer = Adam(global_ac.q2.parameters(), lr=args.lr, eps=1e-4)
    cpc_optimizer = Adam(global_cpc.parameters(), lr=args.lr, eps=1e-4)
    alpha_optim = Adam([global_ac.log_alpha], lr=args.lr, eps=1e-4)

    # Prepare for interaction with environment
    o, ep_ret, ep_len = env.reset(), 0, 0
    if args.cpc:
        c_hidden = global_cpc.init_hidden(1, args.c_dim, use_gpu=args.cuda)
        c1, c_hidden = global_cpc.predict(o, c_hidden)
        assert len(c1.shape) == 3
        c1 = c1.flatten().cpu().numpy()
        all_embeddings = []
        meta = []
    trajectory = list()
    p2 = env.p2
    p2_list = [str(p2)]
    discard = False
    uncertainties = []
    local_t, local_e = 0, 0
    t = T.value()
    e = E.value()
    glod_input = list()
    glod_target = list()
    # Main loop: collect experience in env and update/log each epoch
    while e <= args.episode:
        with torch.no_grad():
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy.
            if t > args.start_steps:
                if args.cpc:
                    a = local_ac.get_action(np.concatenate((o, c1), axis=0), device=device)
                    a_prob = local_ac.act(
                        torch.as_tensor(np.expand_dims(np.concatenate((o, c1), axis=0), axis=0), dtype=torch.float32,
                                        device=device))
                else:
                    a = local_ac.get_action(o, greedy=True, device=device)
                    a_prob = local_ac.act(torch.as_tensor(np.expand_dims(o, axis=0), dtype=torch.float32, device=device))
            else:
                a = env.action_space.sample()
                a_prob = local_ac.act(torch.as_tensor(np.expand_dims(o, axis=0), dtype=torch.float32, device=device))
        uncertainty = ood_scores(a_prob).item()

        # Step the env
        o2, r, d, info = env.step(a)
        if info.get('no_data_receive', False):
            discard = True
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if (ep_len == args.max_ep_len) or discard else d
        glod_input.append(o), glod_target.append(a)

        if args.cpc:
            # changed the trace structure for further analysis
            c2, c_hidden = global_cpc.predict(o2, c_hidden)
            assert len(c2.shape) == 3
            c2 = c2.flatten().cpu().numpy()
            replay_buffer.store(np.concatenate((o, c1), axis=0), a, r, np.concatenate((o2, c2), axis=0), d)
            trajectory.append([o, a, r, o2, d, c1, c2, ep_len])
            all_embeddings.append(c1)
            meta.append([env.p2, local_e, ep_len, r, a, uncertainty])
            c1 = c2
            trajectory.append([o, a, r, o2, d, c1, c2])
        else:
            replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2
        T.increment()
        t = T.value()
        local_t += 1

        # End of trajectory handling
        if d or (ep_len == args.max_ep_len) or discard:
            replay_buffer.store(trajectory)
            E.increment()
            e = E.value()
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
                "Process {}, opponent:{}, # of global_episode :{},  # of global_steps :{}, round score: {}, mean score : {:.1f}, win_rate:{}, steps: {}, alpha: {}".format(
                    rank, args.p2, e, t, ep_ret, m_score, win_rate, ep_len, alpha))
            writer.add_scalar("metrics/round_score", ep_ret, e)
            writer.add_scalar("metrics/mean_score", m_score.item(), e)
            writer.add_scalar("metrics/win_rate", win_rate.item(), e)
            writer.add_scalar("metrics/round_step", ep_len, e)
            writer.add_scalar("metrics/alpha", alpha, e)

            # CPC update handing
            if local_e > args.batch_size and local_e % args.update_every == 0 and args.cpc:
                data, indexes, min_len = replay_buffer.sample_traj(args.batch_size)
                global_cpc.train()
                cpc_optimizer.zero_grad()
                c_hidden = global_cpc.init_hidden(len(data), args.c_dim, use_gpu=args.cuda)
                acc, loss, latents = global_cpc(data, c_hidden)

                replay_buffer.update_latent(indexes, min_len, latents.detach())
                loss.backward()
                # add gradient clipping
                nn.utils.clip_grad_norm_(global_cpc.parameters(), 20)
                cpc_optimizer.step()

                writer.add_scalar("training/acc", acc, e)
                writer.add_scalar("training/cpc_loss", loss.detach().item(), e)

                all_embeddings = np.array(all_embeddings)
                writer.add_embedding(mat=all_embeddings, metadata=meta, metadata_header=["opponent", "round", "step", "reward", "action","uncertainty"])
                c_hidden = global_cpc.init_hidden(1, args.c_dim, use_gpu=args.cuda)
            o, ep_ret, ep_len = env.reset(), 0, 0
            trajectory = list()
            discard = False

        # OOD update stage
        if (t >= args.ood_update_step and local_t % args.ood_update_step == 0 or replay_buffer.is_full()) and args.ood:
            # used all the data collected from the last args.ood_update_steps as the train data
            print("Conduct OOD updating")
            ood_train = (glod_input, glod_target)
            glod_model = convert_to_glod(global_ac.pi,train_loader=ood_train, hidden_dim=args.hid, act_dim=act_dim, device=device)
            glod_scores = retrieve_scores(glod_model, replay_buffer.obs_buf[:replay_buffer.size], device=device, k=args.ood_K)
            glod_scores = glod_scores.detach().cpu().numpy()
            print(len(glod_scores))
            writer.add_histogram(values=glod_scores, max_bins=300, global_step=local_t, tag="OOD")
            drop_points = np.percentile(a=glod_scores, q=[args.ood_drop_lower, args.ood_drop_upper])
            lower, upper = drop_points[0], drop_points[1]
            print(lower,upper)
            mask = np.logical_and((glod_scores >= lower), (glod_scores <= upper))
            reserved_indexes = np.argwhere(mask).flatten()
            print(len(reserved_indexes))
            if len(reserved_indexes) > 0:
                replay_buffer.ood_drop(reserved_indexes)
                glod_input = list()
                glod_target = list()

        # SAC Update handling
        if local_t >= args.update_after and local_t % args.update_every == 0:
            for j in range(args.update_every):

                batch = replay_buffer.sample_trans(batch_size=args.batch_size,device=device)
                # First run one gradient descent step for Q1 and Q2
                q1_optimizer.zero_grad()
                q2_optimizer.zero_grad()
                loss_q = local_ac.compute_loss_q(batch, global_ac_targ, args.gamma, alpha)
                loss_q.backward()

                # Next run one gradient descent step for pi.
                pi_optimizer.zero_grad()
                loss_pi, entropy = local_ac.compute_loss_pi(batch, alpha)
                loss_pi.backward()

                alpha_optim.zero_grad()
                alpha_loss = -(local_ac.log_alpha * (entropy + target_entropy).detach()).mean()
                alpha_loss.backward(retain_graph=False)
                alpha = max(local_ac.log_alpha.exp().item(), args.min_alpha) if not args.fix_alpha else args.min_alpha

                nn.utils.clip_grad_norm_(local_ac.parameters(), 20)
                for global_param, local_param in zip(global_ac.parameters(), local_ac.parameters()):
                    global_param._grad = local_param.grad

                pi_optimizer.step()
                q1_optimizer.step()
                q2_optimizer.step()
                alpha_optim.step()

                state_dict = global_ac.state_dict()
                local_ac.load_state_dict(state_dict)

                # Finally, update target networks by polyak averaging.
                with torch.no_grad():
                    for p, p_targ in zip(global_ac.parameters(), global_ac_targ.parameters()):
                        p_targ.data.copy_((1 - args.polyak) * p.data + args.polyak * p_targ.data)

                writer.add_scalar("training/pi_loss", loss_pi.detach().item(), t)
                writer.add_scalar("training/q_loss", loss_q.detach().item(), t)
                writer.add_scalar("training/alpha_loss", alpha_loss.detach().item(), t)
                writer.add_scalar("training/entropy", entropy.detach().mean().item(), t)

        if t % args.save_freq == 0 and t > 0:
            torch.save(global_ac.state_dict(), os.path.join(args.save_dir, args.exp_name, args.model_para))
            torch.save(global_cpc.state_dict(), os.path.join(args.save_dir, args.exp_name, args.cpc_para))
            state_dict_trans(global_ac.state_dict(), os.path.join(args.save_dir, args.exp_name,  args.numpy_para))
            torch.save((e, t, list(scores), list(wins)), os.path.join(args.save_dir, args.exp_name, args.train_indicator))
            print("Saving model at episode:{}".format(t))