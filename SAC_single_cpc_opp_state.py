import gym
import os
import time
import itertools
import numpy as np
import torch
from games import Soccer,SoccerPLUS
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from Policy_New import Policy
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from copy import deepcopy
from OppModeling.ReplayBuffer import ReplayBufferOppo
from OppModeling.CPC import CPC
from OppModeling.utils import Counter
from OppModeling.logger import get_logger

def _adjust_learning_rate(optimiser, lr):
    for param_group in optimiser.param_groups:
        param_group['lr'] = lr

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, output_activation=nn.Identity):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, output_activation)

    def forward(self, obs):
        net_out = self.net(obs)
        a_prob = F.softmax(net_out, dim=1).clamp(min=1e-20, max=1-1e-20)
        return a_prob

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, output_activation=nn.Identity):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, output_activation)

    def forward(self, obs):
        net_out = self.net(obs)
        return net_out


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.LeakyReLU):
        super().__init__()

        obs_dim = observation_space
        act_dim = action_space
        # act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = Actor(obs_dim, act_dim, hidden_sizes, activation)
        self.q1 = Critic(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = Critic(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, greedy=False):
        with torch.no_grad():
            a_prob = self.pi(obs)
            return a_prob


def sac(env_fn, actor_critic=MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, policy_type = 1,
        logger_kwargs=dict(), save_freq=1000, save_dir=None):

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    opp_policy = Policy(game=env, player_num=False)
    test_env = SoccerPLUS(visual=False)
    test_opp_policy = Policy(game=test_env, player_num=False)
    obs_dim = env.n_features
    act_dim = env.n_actions #env.n_actions

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    # act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(obs_dim, act_dim, **ac_kwargs)
    ac_targ = deepcopy(ac)
    if torch.cuda.is_available():
        ac.cuda()
        ac_targ.cuda()

    device = torch.device('cuda')
    if args.cpc:
        cpc = CPC(timestep=args.timestep, obs_dim=4, hidden_sizes=[args.hid] * args.l, z_dim=args.z_dim,
                         c_dim=args.c_dim, device=device)
    else:
        cpc = None

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)

    # Experience buffer
    T = Counter()  # training step
    E = Counter()  # training episode

    replay_buffer = ReplayBufferOppo(obs_dim=obs_dim, max_size=args.replay_size, cpc=args.cpc,
                                    cpc_model=cpc, writer=writer_cpc,T=T)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.q1, ac.q2])

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q1_optimizer = Adam(ac.q1.parameters(), lr=lr)
    q2_optimizer = Adam(ac.q2.parameters(), lr=lr)
    if args.cpc:
        cpc_optimizer = Adam(cpc.parameters(), lr=args.lr, eps=1e-4)

    # Set up model saving

    # product action
    def get_actions_info(a_prob):
        a_dis = Categorical(a_prob)
        max_a = torch.argmax(a_prob)
        sample_a = a_dis.sample().cpu()
        z = a_prob == 0.0
        z = z.float() * 1e-20
        log_a_prob = torch.log(a_prob + z)
        return a_prob, log_a_prob, sample_a, max_a

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a_prob, log_a_prob, sample_a, max_a = get_actions_info(ac.pi(o2))

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2)
            q2_pi_targ = ac_targ.q2(o2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * torch.sum(a_prob * (q_pi_targ - alpha * log_a_prob),dim=1)

        # MSE loss against Bellman backup
        q1 = ac.q1(o).gather(1, a.unsqueeze(-1).long())
        q2 = ac.q2(o).gather(1, a.unsqueeze(-1).long())
        loss_q1 = F.mse_loss(q1, backup.unsqueeze(-1))
        loss_q2 = F.mse_loss(q2, backup.unsqueeze(-1))
        loss_q = loss_q1 + loss_q2

        return loss_q

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        a_prob, log_a_prob, sample_a, max_a = get_actions_info(ac.pi(o))
        q1_pi = ac.q1(o)
        q2_pi = ac.q2(o)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = torch.sum(a_prob * (alpha * log_a_prob - q_pi),dim=1,keepdim=True).mean()
        entropy = torch.sum(log_a_prob * a_prob, dim=1).detach()

        # Useful info for logging
        pi_info = dict(LogPi=entropy.cpu().numpy())
        return loss_pi, entropy

    def update():
        data = replay_buffer.sample_trans(args.batch_size, device=device)
        # First run one gradient descent step for Q1 and Q2
        q1_optimizer.zero_grad()
        q2_optimizer.zero_grad()
        loss_q = compute_loss_q(data)
        loss_q.backward()
        nn.utils.clip_grad_norm_(ac.parameters(), max_norm=10, norm_type=2)
        q1_optimizer.step()
        q2_optimizer.step()

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, entropy = compute_loss_pi(data)
        loss_pi.backward()
        nn.utils.clip_grad_norm_(ac.parameters(), max_norm=10, norm_type=2)
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        # for p in q_params:
            # p.requires_grad = True

        # Record things

        if t >= update_after:
            # lr = max(args.lr * 2 ** (-(t-update_after) * 0.0001), 1e-10)
            _adjust_learning_rate(q1_optimizer, max(lr, 1e-10))
            _adjust_learning_rate(q2_optimizer, max(lr, 1e-10))
            _adjust_learning_rate(pi_optimizer, max(lr, 1e-10))

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                p_targ.data.copy_((1 - polyak) * p.data + polyak * p_targ.data)

        writer.add_scalar("training/pi_loss", loss_pi.detach().item(), t)
        writer.add_scalar("training/q_loss", loss_q.detach().item(), t)
        writer.add_scalar("training/entropy", entropy.detach().mean().item(), t)
        writer.add_scalar("training/lr", lr, t)

    def update_cpc():
        data, indexes, min_len = replay_buffer.sample_traj(args.cpc_batch)
        data = data[:,:,3:]
        cpc_optimizer.zero_grad()
        c_hidden = cpc.init_hidden(len(data), args.c_dim)
        acc, loss, latents = cpc(data, c_hidden)

        # replay_buffer.update_latent(indexes, min_len, latents.detach())
        loss.backward()
        # add gradient clipping
        nn.utils.clip_grad_norm_(cpc.parameters(), max_norm=20, norm_type=2)
        cpc_optimizer.step()
        writer_cpc.add_scalar("learner/cpc_acc", acc, t)
        writer_cpc.add_scalar("learner/cpc_loss", loss.detach().item(), t)

    def get_action(o, greedy=False):
        if len(o.shape) == 1:
            o = np.expand_dims(o, axis=0)
        a_prob = ac.act(torch.as_tensor(o, dtype=torch.float32,device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")), greedy)
        a_prob, log_a_prob, sample_a, max_a = get_actions_info(a_prob)
        action = sample_a if not greedy else max_a
        return action.item()

    def get_opp_policy(p):
        p_sample = np.random.rand()
        if p_sample < p:
            return args.opp1
        else:
            return args.opp2
    def test_agent(epoch, t_opp, writer):
        if num_test_episodes == 0:
            return
        with torch.no_grad():
            win = 0
            total_ret = 0
            total_len = 0
            for j in range(num_test_episodes):
                o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
                while not (d or (ep_len == max_ep_len)):
                    # Take deterministic actions at test time
                    o2, r, d, _ = test_env.step(get_action(o, True), test_opp_policy.get_actions(t_opp))
                    r *= 10
                    # test_env.render()
                    o = o2
                    ep_ret += r
                    ep_len += 1
                total_ret += ep_ret
                total_len += ep_len
                if(ep_ret == 50):
                    win += 1
            mean_score = total_ret / num_test_episodes
            win_rate = win / num_test_episodes
            mean_len = total_len/ num_test_episodes
            print("opponent:\t{}\ntest epoch:\t{}\nmean score:\t{:.1f}\nwin_rate:\t{}\nmean len:\t{}".format(
    t_opp, epoch, mean_score, win_rate, mean_len))
            writer.add_scalar("test/mean_score", mean_score, epoch)
            writer.add_scalar("test/win_rate", win_rate, epoch)
            writer.add_scalar("test/mean_len", mean_len,epoch)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    scores = []
    trajectory, meta = [], []
    o, ep_ret, ep_len = env.reset(), 0, 0
    discard = False
    opp = get_opp_policy(args.p1)

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        T.increment()

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        with torch.no_grad():
            if t >= start_steps:
                a = get_action(o)
            else:
                a = np.random.randint(act_dim)


        # Step the env
        o2, r, d, info = env.step(a,opp_policy.get_actions(opp))
        if info.get('no_data_receive', False):
            discard = True
        env.render()
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len or discard else d

        # Store experience to replay buffer
        # replay_buffer.store(o, a, r, o2, d)
        e = E.value()
        transition = (o, a, r, o2, d)
        trajectory.append(transition)
        meta.append([opp, 1, e, ep_len, r, a])

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len) or discard:
            scores.append(ep_ret)
            logger.info("total_step:{}, total_episode:{}, opp:{}, round len:{}, round score:{}, 100 mean score:{}， 10 mean Score:{}".format(t, e, opp, ep_len, ep_ret, np.mean(scores[-100:]),np.mean(scores[-10:])))
            writer.add_scalar("metrics/round_score", ep_ret, t)
            writer.add_scalar("metrics/round_step", ep_len, t)
            writer.add_scalar("metrics/alpha", alpha, t)
            o, ep_ret, ep_len = env.reset(), 0, 0
            replay_buffer.store(trajectory, meta=meta)
            trajectory, meta = [], []
            E.increment()
            if t <= args.change_step:
                opp = get_opp_policy(args.p1)
            else:
                opp = get_opp_policy(args.p2)
            discard = False


        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                update()

        # CPC update handing
        if args.cpc and e > args.cpc_batch * 2 and e % args.cpc_update_freq  == 0:
            for _ in range(args.cpc_update_freq):
                update_cpc()

        if t >= update_after and t % save_freq == 0:

            # Test the performance of the deterministic version of the agent.
            test_agent(t, args.opp1, writer_1)
            test_agent(t, args.opp2, writer_3)

        # End of epoch handling
        # if t >= update_after and t % 10000 == 0:
        #     torch.save(ac.state_dict(), os.path.join(save_dir, str(t) + "_model"))
        #     print("Saving model at episode:{}".format(t))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default="FightingiceDataFrameskip-v0")
    # parser.add_argument('--p2', type=str, default="Toothless")
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--policy_type', type=int, default=1)
    parser.add_argument('--replay_size', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('--save_dir', type=str, default='NewPolicy/SAC/')

    parser.add_argument('--p1', type=float, default=0.5)
    parser.add_argument('--p2', type=float, default=1.0)
    parser.add_argument('--opp1', type=int, default=6)
    parser.add_argument('--opp2', type=int, default=7)
    parser.add_argument('--test_episodes', type=int, default=50)
    parser.add_argument('--save_freq', type=int, default=500)
    parser.add_argument('--change_step', type=int, default=100000)

    # CPC setting
    parser.add_argument('--cpc', default=False, action="store_true")
    parser.add_argument('--cpc_batch', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=32)
    parser.add_argument('--c_dim', type=int, default=16)
    parser.add_argument('--timestep', type=int, default=10)
    parser.add_argument('--cpc_update_freq', type=int, default=1, )
    parser.add_argument('--forget_percent', type=float, default=0.2, )
    
    args = parser.parse_args()

    save_dir = os.path.join(args.save_dir, str(args.opp1) + '_' + str(args.opp2), str(args.p1) + '_' + str(args.p2) + '_' +str(args.seed) + '_' + str(args.cpc_batch))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tensorboard_dir = os.path.join(save_dir, "runs")
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    tensorboard_cpc_dir = os.path.join(save_dir, "cpc")
    if not os.path.exists(tensorboard_cpc_dir):
        os.makedirs(tensorboard_cpc_dir)
    writer_cpc = SummaryWriter(log_dir=tensorboard_cpc_dir)

    test_save_dir = os.path.join(args.save_dir, str(args.opp1) + '_' + str(args.opp2), "test_" + str(args.seed))
    tensorboard_dir_1 = os.path.join(test_save_dir, "test_" + str(args.opp1) + "_" + str(args.p1) + '_' + str(args.p2) + '_' + str(args.seed))
    tensorboard_dir_3 = os.path.join(test_save_dir, "test_" + str(args.opp2) + "_" + str(args.p1) + '_' + str(args.p2) + '_' + str(args.seed))
    if not os.path.exists(tensorboard_dir_1):
        os.makedirs(tensorboard_dir_1)
    if not os.path.exists(tensorboard_dir_3):
        os.makedirs(tensorboard_dir_3)
    writer_1 = SummaryWriter(log_dir=tensorboard_dir_1)
    writer_3 = SummaryWriter(log_dir=tensorboard_dir_3)

    filename = save_dir + '_exp.log'
    if not os.path.isfile(filename):
        f = open(filename,mode = 'w')
        f.close()
    logger = get_logger(filename)

    argument_file = save_dir + '.args'
    argsDict = args.__dict__
    with open(argument_file, 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------' + '\n')

    torch.set_num_threads(torch.get_num_threads())

    sac(lambda: SoccerPLUS(visual=False), actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs, policy_type=args.policy_type,  replay_size=args.replay_size,
        lr=args.lr, alpha=args.alpha, batch_size=args.batch_size, start_steps=10000, steps_per_epoch=1000, polyak=0.995,
        update_after=10000, update_every=1, num_test_episodes=args.test_episodes, max_ep_len=1000, save_freq=500,
        logger_kwargs=dict(), save_dir=save_dir)