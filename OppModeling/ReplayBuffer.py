import torch
import random
import numpy as np
import multiprocessing as mp
from OppModeling.utils import combined_shape
from OppModeling.DTW import accelerated_dtw
from OppModeling.Fast_DTW import fastdtw
import os
import torch
import torch.nn.functional as F
import numpy as np
import sklearn.cluster as sc
from torch.utils.tensorboard import SummaryWriter
import collections
import time
# from sklearn.manifold import TSNE
# import pandas as pd
# import matplotlib.pyplot as plt


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
    def __init__(self, max_size, obs_dim, cpc=False, forget_percent=0.2, cpc_model=None, writer=None,E=None,T=None):
        self.trajectories = list()
        self.latents = list()
        self.traj_len = list()
        self.meta = list()
        self.obs_dim = obs_dim
        self.size = 0
        self.E = E
        self.writer = writer
        self.cpc = cpc
        self.cpc_model = cpc_model
        self.forget_percent = forget_percent
        self.max_size = max_size
        self.save_e = 0
        self.num6 = 0
        self.num7 = 0
        self.T = T

    def store(self, trajectory, latents=None, meta=None):
        while self.size + len(trajectory) >= self.max_size:
            self.forget(len(trajectory))
        self.trajectories.append(trajectory)
        self.traj_len.append(len(trajectory))
        self.meta.append(meta)
        if meta[0][0] == 6:
            self.num6 += 1
        else:
            self.num7 += 1
        # print(self.num6, self.num7)
        self.writer.add_scalar('buffer/num6', self.num6, self.T.value())
        self.writer.add_scalar('buffer/num7', self.num7, self.T.value())
        self.writer.add_scalar('buffer/ratio', self.num6 / (self.num6 + self.num7), self.T.value())
        self.size += len(trajectory)

    def forget(self, trajectory_length):
        if not self.cpc:
            self.trajectories.pop(0)
            self.size -= self.traj_len.pop(0)
            if self.meta[0][0][0] == 6:
                self.num6 -= 1
            else:
                self.num7 -= 1
            self.meta.pop(0)
        else:
            # labels = self.create_latents(self.T.value())
            # labels = labels.tolist()

            # while self.size + trajectory_length >= self.max_size:
            #     nmax = 0
            #     lmax = 0
            #     N=max(labels)+1
            #     for i in range(N):
            #         ll=labels.count(i)
            #         if labels.count(i)>lmax:
            #             nmax=i
            #             lmax=ll

            #     index = 0
            #     # print(labels)
            #     # print(collections.Counter(labels))
            #     # print(nmax)
            #     for i in range(len(labels)):
            #         if labels[i] == nmax:
            #             index = i
            #             # print(index)
            #             break

            #     if self.meta[index][0][0] == 6:
            #         self.num6 -= 1
            #     else:
            #         self.num7 -= 1
            #     # print(self.num6, self.num7)
            #     self.trajectories.pop(index)
            #     self.meta.pop(index)
            #     self.size -= self.traj_len.pop(index)
            #     labels.pop(index)

            self.create_latents(self.T.value())
            self.trajectories.pop(0)
            self.meta.pop(0)
            self.size -= self.traj_len.pop(0)

            # self.create_latents(self.E.value())
            # distance_matrix = self.latent_distance()
            # closest_k = int(len(self.latents) * self.forget_percent)
            # ind =np.argpartition(distance_matrix, closest_k, axis=1)[:,:closest_k]
            # kmean_dis = distance_matrix.take(ind).mean(axis=1)
            # remove_index = np.argpartition(kmean_dis, closest_k, axis=-1)[:closest_k]
            # remove_index = np.sort(remove_index)[::-1]
            # for index in remove_index:
            #     del self.trajectories[index]
            #     del self.latents[index]
            #     del self.meta[index]
            #     self.size -= self.traj_len.pop(index)

    def create_latents(self, e):
        assert self.cpc_model is not None
        assert self.writer is not None
        latents = list()
        all_embeddings = list()
        for trajectory in self.trajectories:
            obs = [tran[0] for tran in trajectory]
            
            obs = np.array(obs)[:, 3:]

            c_hidden = self.cpc_model.init_hidden(1, self.cpc_model.c_dim)
            traj_c, c_hidden = self.cpc_model.predict(obs, c_hidden)
            latents.append(traj_c.squeeze())
            all_embeddings += traj_c.squeeze().tolist()
        # if want to directly delete the elements, latents need to be a list
        self.latents = latents
        # labels = self.AgglomerativeClustering()
        flat_meta = list()
        for meta_traj in self.meta:
            for meta in meta_traj:
                flat_meta.append(meta)
        # self.writer.add_embedding(mat=np.array(all_embeddings), metadata=flat_meta,global_step=e,
        #              metadata_header=["opponent", "rank", "round", "step", "reward", "action", ])
        
        # if e - self.save_e >= 2000:
        #     self.save_e = e
        #     self.writer.add_embedding(mat=np.array(all_embeddings), metadata=flat_meta,global_step=e,
        #                          metadata_header=["opponent", "rank", "round", "step", "reward", "action", ])
        # return labels


    def AgglomerativeClustering(self):
        cnt = 0
        for i in self.latents:
            cnt += len(i)
        mean_len = cnt // len(self.latents)
        interpolate_t = torch.zeros(len(self.latents), 16, mean_len).cuda() #NxCxT
        for i in range(len(self.latents)):
            t = self.latents[i].t().unsqueeze(0) # 1xCxT
            t = F.interpolate(t, mean_len, mode='linear', align_corners=False)
            interpolate_t[i] = t
        interpolate = interpolate_t

        t_data = interpolate_t
        l2_d = [[0 for _ in range(len(t_data))] for _ in range(len(t_data))]
        for i in range(len(t_data)):
            for j in range(i):
                l2_d[i][j] = l2_d[j][i] = F.pairwise_distance(t_data[i].unsqueeze(0), t_data[j].unsqueeze(0), p=2).sum().item()

        l2_d_mean = np.mean(l2_d)
        
        labels = sc.AgglomerativeClustering(affinity='precomputed', n_clusters=None,distance_threshold=l2_d_mean,linkage='average').fit_predict(l2_d)
        return labels



    # def latent_distance(self):
    #     distance_matrix = np.empty((len(self.latents), len(self.latents),))
    #     distance_matrix[:] = np.nan
    #     seq = [(self.latents[i], self.latents[j]) for i in range(len(self.latents)) for j in range(i)]

    #     with mp.Pool(4) as p:
    #         distances = p.starmap(worker, seq)

    #     index = 0
    #     for i in range(len(self.latents)):
    #         for j in range(i+1):
    #             if np.isnan(distance_matrix[i][j]):
    #                 if i == j:
    #                     distance_matrix[i][j] = 0
    #                 else:
    #                     distance_matrix[i][j] = distance_matrix[j][i] = distances[index]
    #                     index += 1
    #     return distance_matrix

    def sample_trans(self, batch_size, device=None):
        if batch_size == 0:
            return dict(obs=torch.as_tensor([], dtype=torch.float32).to(device), obs2=torch.as_tensor([], dtype=torch.float32).to(device), 
                act=torch.as_tensor([], dtype=torch.float32).to(device), rew=torch.as_tensor([], dtype=torch.float32).to(device), done=torch.as_tensor([], dtype=torch.float32).to(device))
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
        batch_size *= 2
        indexes = np.random.randint(len(self.trajectories), size=batch_size)
        mean_len = np.median([self.traj_len[i] for i in indexes])
        indexes = [i for i in indexes if self.traj_len[i] >= mean_len]
        min_len = min([self.traj_len[i] for i in indexes])
        # cut off using the min length
        batch = []
        for i in indexes:
            # each sampled trace obs len = min_len + 1(the last o2 in this trace)
            batch.append([self.trajectories[i][j][0] for j in range(min_len)] + [self.trajectories[i][min_len-1][3]])
        batch = np.array(batch, dtype=np.float)
        assert batch.shape == (len(indexes), min_len + 1, self.obs_dim)
        return batch, indexes, min_len

    # currently can only update the trans index less than min
    def update_latent(self, indexes, min_len, latents):
        for i, index in enumerate(indexes):
            for j in range(min_len):
                self.trajectories[index][j][-2] = latents[i][j].cpu().numpy()  # update c1
                self.trajectories[index][j][-1] = latents[i][j+1].cpu().numpy()   # update c2
        print("updated latents")

    def is_full(self):
        return self.size >= self.max_size

    def __len__(self):
        return self.size

    def load_factor(self):
        return self.size / self.max_size


def worker(latent1, latent2):
    dis, _, _, _ = accelerated_dtw(latent1, latent2, dist="euclidean")
    return dis

#slower
def worker_(latent1, latent2):
    dis, _ = fastdtw(latent1, latent2)
    return dis