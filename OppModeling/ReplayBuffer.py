import torch
import random
import numpy as np
from OppModeling.utils import combined_shape


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, size):
        self.obs_dim = obs_dim
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.p2_buf = np.array([None for i in range(size)])
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, p2=None):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.p2_buf[self.ptr] = p2
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def store_trajectory(self, trajectory):
        for i in trajectory:
            self.store(i["obs"], i["action"], i["reward"], i["next_obs"], i["done"])

    def sample_trans(self, batch_size=32, device=None):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in batch.items()}

    def reset(self):
        self.__init__(self.obs_dim,self.size)

    def ood_drop(self, reserved_indexes):
        print("before ood drop: {}".format(self.size))
        reserved_len = len(reserved_indexes)
        self.obs_buf[:reserved_len] = self.obs_buf[reserved_indexes]
        self.obs_buf[reserved_len:] = 0
        self.obs2_buf[:reserved_len] = self.obs2_buf[reserved_indexes]
        self.obs2_buf[reserved_len:] = 0
        self.act_buf[:reserved_len] = self.act_buf[reserved_indexes]
        self.act_buf[reserved_len:] = 0
        self.rew_buf[:reserved_len] = self.rew_buf[reserved_indexes]
        self.rew_buf[reserved_len:] = 0
        self.done_buf[:reserved_len] = self.done_buf[reserved_indexes]
        self.done_buf[reserved_len:] = 0
        self.p2_buf[:reserved_len] = self.p2_buf[reserved_indexes]
        self.p2_buf[reserved_len:] = 0
        self.size = min(reserved_len, self.max_size)
        self.ptr = self.size % self.max_size
        print("after ood drop: {}".format(self.size))

    def is_full(self):
        return self.size == self.max_size




class ReplayBufferShare:
    """
    A simple FIFO experience replay buffer for shared memory.
    """

    def __init__(self, buffer, size):
        self.buffer = buffer
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append(dict(obs=obs, next_obs=next_obs, action=act, reward=rew, done=done))
        else:
            self.buffer.pop(0)
            self.buffer.append(dict(obs=obs, next_obs=next_obs, action=act, reward=rew, done=done))
        self.ptr = (self.ptr + 1) % self.max_size

    def sample_batch(self, batch_size=32, device=None):
        idxs = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in idxs]
        obs_buf, obs2_buf, act_buf, rew_buf, done_buf = [], [], [], [], []
        for trans in batch:
            obs_buf.append(trans["obs"])
            obs2_buf.append(trans["next_obs"])
            act_buf.append(trans["action"])
            rew_buf.append(trans["reward"])
            done_buf.append(trans["done"])
        batch_dict = dict(obs=obs_buf, obs2=obs2_buf, act=act_buf, rew=rew_buf, done=done_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in batch_dict.items()}


class ReplayBufferOppo:
    def __init__(self, max_size, obs_dim, cpc=False):
        self.trajectories = list()
        self.latents = list()
        self.traj_len = list()
        self.meta = list()
        self.obs_dim = obs_dim
        self.size = 0
        self.cpc = cpc
        self.max_size = max_size

    def store(self, trajectory, latents=None, meta=None):
        while self.size + len(trajectory) >= self.max_size:
            self.forget()
        self.trajectories.append(trajectory)
        self.traj_len.append(len(trajectory))
        if self.cpc:
            self.meta.append(meta)
            self.latents.append(latents)
        self.size += len(trajectory)

    def forget(self):
        if not self.cpc:
            self.trajectories.pop(0)
            self.size -= self.traj_len.pop(0)
        else:
            # TODO add cpc forget functions here
            self.trace_distance()
            pass

    def trace_distance(self):
        pass

    def sample_trans(self, batch_size, device=None):
        indexes = np.arange(len(self.trajectories))
        prob = np.array(self.traj_len) / sum(self.traj_len)
        sampled_trans = []
        sampled_traj_index = np.random.choice(indexes, size=batch_size, replace=True, p=prob)
        for index in sampled_traj_index:
            sampled_trans.append(random.choice(self.trajectories[index]))
        obs_buf, obs2_buf, act_buf, rew_buf, done_buf = [], [], [], [], []
        for trans in sampled_trans:
            obs_buf.append(trans[0])
            obs2_buf.append(trans[3])
            act_buf.append(trans[1])
            rew_buf.append(trans[2])
            done_buf.append(trans[4])
        batch = dict(obs=obs_buf, obs2=obs2_buf, act=act_buf, rew=rew_buf, done=done_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in batch.items()}

    # This function sample batch of trace for CPC training, only return the batch of obs
    def sample_traj(self, batch_size):
        indexes = np.random.randint(len(self.trajectories), size=batch_size)
        min_len = min([self.traj_len[i] for i in indexes])
        # cut off using the min length
        batch = []
        for i in indexes:
            # each sampled trace obs len = min_len + 1(the last o2 in this trace)
            batch.append([self.trajectories[i][j][0] for j in range(min_len)] + [self.trajectories[i][min_len-1][3]])
        batch = np.array(batch, dtype=np.float)
        assert batch.shape == (batch_size, min_len + 1, self.obs_dim)
        return batch, indexes, min_len

    # currently can only update the trans index less than min
    def update_latent(self, indexes, min_len, latents):
        for i, index in enumerate(indexes):
            for j in range(min_len):
                self.trajectories[index][j][-2] = latents[i][j].cpu().numpy()  # update c1
                self.trajectories[index][j][-1] = latents[i][j+1].cpu().numpy()   # update c2
        print("updated latents")

    def is_full(self):
        return len(self.trajectories) == self.max_size

    def __len__(self):
        return self.size

    def load_factor(self):
        return self.size / self.max_size

