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


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32, device=None):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in batch.items()}


def sac(env_fn, actor_critic=MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=1000, save_dir=None):

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    opp_policy = Policy(game=env, player_num=False)
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

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.q1, ac.q2])

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q1_optimizer = Adam(ac.q1.parameters(), lr=lr)
    q2_optimizer = Adam(ac.q2.parameters(), lr=lr)

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

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info

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

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q1_optimizer.zero_grad()
        q2_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        nn.utils.clip_grad_norm_(ac.parameters(), max_norm=10, norm_type=2)
        q1_optimizer.step()
        q2_optimizer.step()

        # Record things

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        # for p in q_params:
        #     p.requires_grad = False

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

    def get_action(o, greedy=False):
        if len(o.shape) == 1:
            o = np.expand_dims(o, axis=0)
        a_prob = ac.act(torch.as_tensor(o, dtype=torch.float32,device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")), greedy)
        a_prob, log_a_prob, sample_a, max_a = get_actions_info(a_prob)
        action = sample_a if not greedy else max_a
        return action.item()

    # def test_agent():
    #     for j in range(num_test_episodes):
    #         o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
    #         while not (d or (ep_len == max_ep_len)):
    #             # Take deterministic actions at test time
    #             o, r, d, _ = test_env.step(get_action(o, True))
    #             # test_env.render()
    #             ep_ret += r
    #             ep_len += 1
    #         print(ep_ret)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    scores = []
    o, ep_ret, ep_len = env.reset(), 0, 0
    discard = False
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        with torch.no_grad():
            if t >= start_steps:
                a = get_action(o)
            else:
                a = np.random.randint(act_dim)


        # Step the env
        o2, r, d, info = env.step(a,opp_policy.get_actions(1))
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
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len) or discard:
            scores.append(ep_ret)
            print("total_step {},round len:{}, round score: {}, 100 mean score: {}， 10 mean Score: {}".format(t,ep_len, ep_ret, np.mean(scores[-100:]),np.mean(scores[-10:])))
            writer.add_scalar("metrics/round_score", ep_ret, t)
            writer.add_scalar("metrics/round_step", ep_len, t)
            writer.add_scalar("metrics/alpha", alpha, t)
            o, ep_ret, ep_len = env.reset(), 0, 0
            discard = False


        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size,device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
                update(data=batch)



        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            # Save model
            if t % save_freq == 0 and t > 0:
                torch.save(ac.state_dict(), os.path.join(save_dir, "model"))
                print("Saving model at episode:{}".format(t))

            # # Test the performance of the deterministic version of the agent.
            # test_agent()
            #
            # # Log info about epoch
            # logger.log_tabular('Epoch', epoch)
            # logger.log_tabular('EpRet', with_min_and_max=True)
            # logger.log_tabular('TestEpRet', with_min_and_max=True)
            # logger.log_tabular('EpLen', average_only=True)
            # logger.log_tabular('TestEpLen', average_only=True)
            # logger.log_tabular('TotalEnvInteracts', t)
            # logger.log_tabular('Q1Vals', with_min_and_max=True)
            # logger.log_tabular('Q2Vals', with_min_and_max=True)
            # logger.log_tabular('LogPi', with_min_and_max=True)
            # logger.log_tabular('LossPi', average_only=True)
            # logger.log_tabular('LossQ', average_only=True)
            # logger.log_tabular('Time', time.time() - start_time)
            # logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="FightingiceDataFrameskip-v0")
    parser.add_argument('--p2', type=str, default="Toothless")
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('--save_dir', type=str, default='OpenAI/SAC/')
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    tensorboard_dir = os.path.join(args.save_dir, "runs")
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    torch.set_num_threads(torch.get_num_threads())

    sac(lambda: SoccerPLUS(), actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs, steps_per_epoch=1000, replay_size=int(1e6),
        polyak=0.995, lr=args.lr, alpha=0.1, batch_size=4096, start_steps=10000,
        update_after=10000, update_every=1, num_test_episodes=5, max_ep_len=1000,save_freq=100,
        logger_kwargs=dict(), save_dir=args.save_dir)