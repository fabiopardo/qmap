import gym
from itertools import product
import numpy as np
import os

from qmap.csv_logger import CSVLogger


class EpisodeRenderer(gym.Wrapper):
    def __init__(self, env, rendering_freq=1, finished_only=False, fast=False, viewer_title='simulation'):
        super(EpisodeRenderer, self).__init__(env)
        self.rendering_freq = rendering_freq
        self.finished_only = finished_only
        self.episodes = 0
        self.env.unwrapped.fast_rendering = fast
        self.env.unwrapped.viewer_title = viewer_title

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if self.rendering_freq != 0 and self.episodes%self.rendering_freq == 0:
            if done or not self.finished_only:
                self.env.render()
        return observation, reward, done, info

    def reset(self):
        self.episodes += 1
        observation = self.env.reset()
        if self.rendering_freq != 0 and self.episodes%self.rendering_freq == 0:
            self.env.render()
        return observation

class PerfLogger(gym.Wrapper):
    def __init__(self, env, gamma, path):
        super(PerfLogger, self).__init__(env)
        self.gamma = gamma
        self.episodes = 0
        self.steps = 0
        self.discount = 1
        self.ep_length = 0
        self.needs_reset = True
        self.logger = CSVLogger(['steps', 'undiscounted return', 'discounted return', 'episode length'], path + '/score')

    def step(self, action):
        assert not self.needs_reset, 'The environment should be reset.'
        self.steps += 1
        self.ep_length += 1
        observation, reward, done, info = self.env.step(action)
        self.undiscounted_return += reward
        self.discounted_return += self.discount * reward
        self.discount *= self.gamma
        if done: self.needs_reset = True
        return observation, reward, done, info

    def reset(self):
        if self.ep_length != 0:
            self.logger.log(self.steps-self.ep_length, self.undiscounted_return, self.discounted_return, self.ep_length)
        self.needs_reset = False
        self.undiscounted_return = 0
        self.discounted_return = 0
        self.ep_length = 0
        self.episodes += 1
        self.discount = 1
        return self.env.reset()

class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps, output_time=False):
        super(TimeLimit, self).__init__(env)
        if output_time:
            self.observation_space += [gym.spaces.Box(np.array([0.0]), np.array([1.0]))]
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        self._output_time = output_time

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._max_episode_steps <= self._elapsed_steps:
            done = True
        if self._output_time:
            observation = observation + [np.array([(self._max_episode_steps-self._elapsed_steps)/self._max_episode_steps])]
        return observation, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        observation = self.env.reset()
        if self._output_time:
            observation += [np.array([1.0])]
        return observation


class TimeInInput(gym.Wrapper):
    def __init__(self, env, max_episode_steps):
        super(TimeInInput, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        self.use_lists = isinstance(env.observation_space, (list,))

        if self.use_lists:
            self.observation_space += [gym.spaces.Box(np.array([-1.0]), np.array([1.0]))]
        else:
            low = env.observation_space.low
            high = env.observation_space.high
            self.observation_space = gym.spaces.Box(np.append(low, -1.0), np.append(high, 1.0))

    def step(self, action):
        assert self._elapsed_steps < self._max_episode_steps
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        remaining_time = ((self._max_episode_steps-self._elapsed_steps)/self._max_episode_steps) * 2 - 1
        if self.use_lists:
            observation = observation + [np.array(remaining_time)] # TODO: find what's best
        else:
            observation = np.append(observation, remaining_time)
        return observation, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        observation = self.env.reset()
        if self.use_lists:
            observation += [np.array([1.0])]
        else:
            observation = np.append(observation, 1.0)
        return observation


# for compatibility with Gym's environments
class ObsList(gym.Wrapper):
    def __init__(self, env):
        super(ObsList, self).__init__(env)
        self.observation_space = [self.env.observation_space]

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return [observation], reward, done, info

    def reset(self):
        observation = self.env.reset()
        return [observation]

# discretize action spaces
class ActionDiscretizer(gym.Wrapper):
    def __init__(self, env, actions_per_dim, min_val=-1.0, max_val=1.0):
        super(ActionDiscretizer, self).__init__(env)
        n_action_dims = env.action_space.shape[0]
        self.actions = np.array(list(product(np.linspace(min_val, max_val, actions_per_dim), repeat=n_action_dims)))
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def step(self, action):
        action = self.actions[action]
        return self.env.step(action)

# merge observation streams
class MergeObservationStreams(gym.Wrapper):
    def __init__(self, env):
        super(MergeObservationStreams, self).__init__(env)
        lows = [obs_space.low for obs_space in env.observation_space]
        highs = [obs_space.high for obs_space in env.observation_space]
        self.observation_space = gym.spaces.Box(np.concatenate(lows), np.concatenate(highs))

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = np.concatenate(observation)
        return observation, reward, done, info

    def reset(self):
        observation = self.env.reset()
        observation = np.concatenate(observation)
        return observation
