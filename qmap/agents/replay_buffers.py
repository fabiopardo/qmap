from baselines.common.schedules import LinearSchedule
from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
import numpy as np
import random


class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, rcw, obs_tp1, done):
        data = (obs_t, action, reward, rcw, obs_tp1, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, rcw, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def _encode_qmap_sample(self, idxes, n_steps=1):
        obses_t, actions, rcws, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i:i+n_steps]
            obs_t = data[0][0]
            action = data[0][1]
            rcw = [d[3] for d in data]
            obs_tp1 = data[-1][4]
            done = [d[5] for d in data]
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rcws.append(np.array(rcw, copy=False))
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(np.array(done, copy=False))
        return np.array(obses_t), np.array(actions), np.array(rcws), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size, time_step):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        weights = np.ones(batch_size)
        return self._encode_sample(idxes) + (weights,)

    def sample_qmap(self, batch_size, time_step, n_steps=1):
        idxes = [random.randint(0, len(self._storage) - n_steps) for _ in range(batch_size)]
        weights = np.ones(batch_size)
        return self._encode_qmap_sample(idxes, n_steps) + (weights,)

    def update_priorities(self, td_errors):
        pass

    def update_priorities_qmap(self, td_errors):
        pass


class DoublePrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha, epsilon, timesteps, initial_p, final_p):
        super(DoublePrioritizedReplayBuffer, self).__init__(size)
        assert alpha > 0
        self._alpha = alpha
        self._epsilon = epsilon
        self._beta_schedule = LinearSchedule(timesteps, initial_p=initial_p, final_p=final_p)
        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

        self._it_sum2 = SumSegmentTree(it_capacity)
        self._it_min2 = MinSegmentTree(it_capacity)
        self._max_priority2 = 1.0

    def add(self, *args, **kwargs):
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

        self._it_sum2[idx] = self._max_priority2 ** self._alpha
        self._it_min2[idx] = self._max_priority2 ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def _sample_proportional2(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum2.sum(0, len(self._storage) - 1)
            idx = self._it_sum2.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, time_step):
        beta = self._beta_schedule.value(time_step)
        assert beta > 0

        idxes = self._sample_proportional(batch_size)
        self.idxes = idxes # keep to update priorities later

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return encoded_sample + (weights,)

    def sample_qmap(self, batch_size, time_step, n_steps=1):
        beta = self._beta_schedule.value(time_step)
        assert beta > 0

        idxes = self._sample_proportional2(batch_size)
        self.idxes2 = idxes # keep to update priorities later

        weights = []
        p_min = self._it_min2.min() / self._it_sum2.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum2[idx] / self._it_sum2.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_qmap_sample(idxes, n_steps)
        return encoded_sample + (weights,)

    def update_priorities(self, td_errors):
        priorities = np.abs(td_errors) + self._epsilon
        idxes = self.idxes
        assert len(idxes) == len(priorities)

        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priority)

    def update_priorities_qmap(self, td_errors):
        priorities = np.abs(td_errors) + self._epsilon
        idxes = self.idxes2
        assert len(idxes) == len(priorities)

        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum2[idx] = priority ** self._alpha
            self._it_min2[idx] = priority ** self._alpha
            self._max_priority2 = max(self._max_priority2, priority)
