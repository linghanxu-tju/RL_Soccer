import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from OppModeling.utils import mlp



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
    def __init__(self,obs_dim, act_dim, hidden_sizes=(256, 256),
                 activation=nn.ReLU):
        super().__init__()

        # build policy and value functions
        self.pi = Actor(obs_dim, act_dim, hidden_sizes, activation)
        self.q1 = Critic(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = Critic(obs_dim, act_dim, hidden_sizes, activation)
        self.log_alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data, ac_targ, gamma, alpha):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a_prob, log_a_prob, sample_a, max_a = self.get_actions_info(self.pi(o2))

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2)
            q2_pi_targ = ac_targ.q2(o2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (a_prob * (q_pi_targ - alpha * log_a_prob)).sum(dim=1)

        # MSE loss against Bellman backup
        q1 = self.q1(o).gather(1, a.unsqueeze(-1).long())
        q2 = self.q2(o).gather(1, a.unsqueeze(-1).long())
        loss_q1 = F.mse_loss(q1, backup.unsqueeze(-1))
        loss_q2 = F.mse_loss(q2, backup.unsqueeze(-1))
        loss_q = loss_q1 + loss_q2

        return loss_q

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data, alpha):
        o = data['obs']
        a_prob, log_a_prob, sample_a, max_a = self.get_actions_info(self.pi(o))
        q1_pi = self.q1(o)
        q2_pi = self.q2(o)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = torch.sum(a_prob * (alpha * log_a_prob - q_pi), dim=1, keepdim=True).mean()
        entropy = torch.sum(log_a_prob * a_prob, dim=1)

        # Useful info for logging
        # pi_info = dict(LogPi=entropy.numpy())
        return loss_pi, entropy.detach()

    def act(self, obs):
        with torch.no_grad():
            a_prob = self.pi(obs)
            return a_prob

    def get_action(self, o, greedy=False, device=None):
        if len(o.shape) == 1:
            o = np.expand_dims(o, axis=0)
        a_prob = self.act(torch.as_tensor(o, dtype=torch.float32, device=device))
        a_prob, log_a_prob, sample_a, max_a = self.get_actions_info(a_prob)
        action = sample_a if not greedy else max_a
        return action.item()

        # product action

    @staticmethod
    def get_actions_info(a_prob):
        a_dis = Categorical(a_prob)
        max_a = torch.argmax(a_prob)
        sample_a = a_dis.sample()
        z = a_prob == 0.0
        z = z.float() * 1e-20
        a_prob += z
        log_a_prob = torch.log(a_prob)
        return a_prob, log_a_prob, sample_a, max_a

