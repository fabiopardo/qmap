from baselines import logger, deepq
import baselines.common.tf_util as U
from baselines.deepq.deepq import ActWrapper
from baselines.deepq.utils import ObservationInput
from agent import Agent
import gym
import numpy as np
import os
import tempfile
import tensorflow as tf
import zipfile


# This agent is intended to match the original OpenAI Baseline's DQN performance and logs

class DQN_Agent(Agent):
    def __init__(self,
        observation_space_shape,
        num_actions,
        q_func,
        replay_buffer,
        exploration,
        lr=5e-4,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        train_freq=1,
        batch_size=32,
        print_freq=100,
        learning_starts=1000,
        gamma=1.0,
        target_network_update_freq=500,
        param_noise=False
    ):
        self.num_actions = num_actions
        self.replay_buffer = replay_buffer
        self.exploration = exploration
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.print_freq = print_freq
        self.learning_starts = learning_starts
        self.target_network_update_freq = target_network_update_freq
        self.param_noise = param_noise

        def make_obs_ph(name):
            return ObservationInput(observation_space_shape, name=name)

        self.act, self.train, self.update_target, _ = deepq.build_train(
            make_obs_ph=make_obs_ph,
            q_func=q_func,
            num_actions=num_actions,
            optimizer=tf.train.AdamOptimizer(learning_rate=lr),
            gamma=gamma,
            grad_norm_clipping=10,
            param_noise=self.param_noise
        )

        U.initialize()
        self.update_target()

        self.t = 0
        self.episode_rewards = []

    def reset(self, ob):
        self.episode_rewards.append(0.0)
        ac = self.choose_action(ob, True)
        self.prev_ob = ob
        self.prev_ac = ac
        self.log()

        return ac

    def step(self, ob, rew, done):
        self.replay_buffer.add(self.prev_ob, self.prev_ac, rew, ob, float(done))
        self.episode_rewards[-1] += rew

        self.optimize()
        if not done:
            ac = self.choose_action(ob, False)
        else:
            ac = None

        self.t += 1
        self.prev_ob = ob
        self.prev_ac = ac

        return ac

    def choose_action(self, ob, reset):
        kwargs = {}
        if not self.param_noise:
            update_eps = self.exploration.value(self.t)
            update_param_noise_threshold = 0.
        else:
            update_eps = 0.
            update_param_noise_threshold = -np.log(1. - self.exploration.value(self.t) + self.exploration.value(self.t) / float(self.num_actions))
            kwargs['reset'] = reset
            kwargs['update_param_noise_threshold'] = update_param_noise_threshold
            kwargs['update_param_noise_scale'] = True
        return self.act(np.array(ob)[None], update_eps=update_eps, **kwargs)[0]

    def optimize(self):
        if self.t > self.learning_starts and self.t % self.train_freq == 0:
            experience = self.replay_buffer.sample(self.batch_size, self.t)
            td_errors = self.train(*experience)
            self.replay_buffer.update_priorities(td_errors)

        if self.t > self.learning_starts and self.t % self.target_network_update_freq == 0:
            self.update_target()

    def log(self):
        if self.print_freq is not None and len(self.episode_rewards) % self.print_freq == 0:
            mean_100ep_reward = round(np.mean(self.episode_rewards[-101:-1]), 1)
            num_episodes = len(self.episode_rewards)
            logger.record_tabular("steps", self.t-1)
            logger.record_tabular("episodes", num_episodes)
            logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
            logger.record_tabular("% time spent exploring", int(100 * self.exploration.value(self.t)))
            logger.dump_tabular()
