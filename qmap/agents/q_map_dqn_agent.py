from baselines import logger, deepq
import baselines.common.tf_util as U
from baselines.deepq.utils import ObservationInput
from datetime import datetime
from gym.utils import seeding
import math
import numpy as np
import os
import tensorflow as tf

from qmap.agents.q_map_renderer import Q_Map_Renderer
from qmap.utils.csv_logger import CSVLogger


class DQN():
    def __init__(self,
        model,
        observation_space,
        n_actions,
        gamma,
        lr,
        replay_buffer,
        batch_size,
        optim_iters,
        grad_norm_clip,
        double_q
    ):
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.optim_iters = optim_iters
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        def make_obs_ph(name):
            return ObservationInput(observation_space, name=name)

        self.act, self.train, self.update_target, _ = deepq.build_train(
            make_obs_ph=make_obs_ph,
            q_func=model,
            num_actions=n_actions,
            optimizer=optimizer,
            gamma=gamma,
            grad_norm_clipping=grad_norm_clip,
            double_q=double_q
        )

    def choose_action(self, ob, stochastic):
        ac = self.act(ob[None], stochastic=False)[0]
        return ac

    def optimize(self, t):
        for iteration in range(self.optim_iters):
            samples = self.replay_buffer.sample(self.batch_size, t)
            obs, acs, rews, obs1, dones, weights = samples
            td_errors = self.train(obs, acs, rews, obs1, dones, weights)
            self.replay_buffer.update_priorities(td_errors)

    def update_target(self):
        self.update_target()


def qmap_build_train(observation_space, coords_shape, model, n_actions, optimizer, grad_norm_clip, scope='q_map'):
    with tf.variable_scope(scope):
        ob_shape = observation_space.shape
        observations = tf.placeholder(tf.float32, [None] + list(ob_shape), name='observations')
        actions = tf.placeholder(tf.int32, [None], name='actions')
        target_qs = tf.placeholder(tf.float32, [None] + list(coords_shape), name='targets')
        weights = tf.placeholder(tf.float32, [None], name='weights')

        q_values = model(inpt=observations, n_actions=n_actions, scope='q_func')
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/q_func")

        target_q_values = model(inpt=observations, n_actions=n_actions, scope='target_q_func')
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/target_q_func")

        action_masks = tf.expand_dims(tf.expand_dims(tf.one_hot(actions, n_actions), axis=1), axis=1)
        qs_selected = tf.reduce_sum(q_values * action_masks, 3)

        td_errors = 1 * (qs_selected - target_qs) # TODO: coefficient?
        losses = tf.reduce_mean(tf.square(td_errors), [1, 2]) # TODO: find best, was U.huber_loss
        weighted_loss = tf.reduce_mean(weights * losses)

        if grad_norm_clip is not None:
            gradients = optimizer.compute_gradients(weighted_loss, var_list=q_func_vars)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, grad_norm_clip), var)
            optimize = optimizer.apply_gradients(gradients)
            grad_norms = [tf.norm(grad) for grad in gradients]
        else:
            optimize = optimizer.minimize(weighted_loss, var_list=q_func_vars)
            grad_norms = None

        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

    errors = tf.reduce_mean(tf.abs(td_errors), [1, 2]) # TODO: try with the losses directly
    compute_q_values = U.function(inputs=[observations], outputs=q_values)
    compute_double_q_values = U.function(inputs=[observations], outputs=[q_values, target_q_values])
    train = U.function(inputs=[observations, actions, target_qs, weights], outputs=errors, updates=[optimize])
    update_target = U.function([], [], updates=[update_target_expr])
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    train_debug = U.function(inputs=[observations, actions, target_qs, weights], outputs=[errors, weighted_loss, grad_norms, trainable_vars], updates=[optimize])

    return compute_q_values, compute_double_q_values, train, update_target, train_debug


class Q_Map():
    def __init__(self,
        model,
        observation_space,
        coords_shape,
        n_actions,
        gamma,
        n_steps,
        lr,
        replay_buffer,
        batch_size,
        optim_iters,
        grad_norm_clip,
        double_q
    ):
        self.coords_shape = coords_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.n_steps = n_steps
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.optim_iters = optim_iters
        self.double_q = double_q
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        self.compute_q_values, self.compute_double_q_values, self.train, self.update_target, _ = qmap_build_train(
            observation_space=observation_space,
            coords_shape=coords_shape,
            model=model,
            n_actions=n_actions,
            optimizer=optimizer,
            grad_norm_clip=grad_norm_clip
        )

    def choose_action(self, ob, goal_rc, q_values=None):
        if q_values is None:
            q_values = self.compute_q_values(ob[None])[0]
        ac = q_values[goal_rc[0], goal_rc[1]].argmax()

        return ac, q_values

    def _optimize(self, obs, acs, rcw1s, obs1, dones, weights):
        if self.double_q:
            next_qs, target_next_qs = self.compute_double_q_values(obs1)
            target_next_acs = target_next_qs.argmax(3)
            mask = np.arange(self.n_actions) == target_next_acs[..., None]
            best_next_qs = next_qs[mask].reshape(target_next_acs.shape)
        else:
            next_qs = self.compute_q_values(obs1)
            best_next_qs = next_qs.max(3)
        clipped_best_next_qs = np.clip(best_next_qs, 0., 1.) # value clipping

        target_qs = clipped_best_next_qs
        window = target_qs.shape[2]
        for i in reversed(range(self.n_steps)):
            target_qs *= self.gamma * (1 - dones[:, i, None, None])
            rows, cols, delta_ws = rcw1s[:, i, 0], rcw1s[:, i, 1], rcw1s[:, i, 2]
            target_qs[np.arange(self.batch_size), rows, cols] = 1

            # take the movement of the window into account
            for j in range(self.batch_size):
                if delta_ws[j] < 0:
                    target_qs[j, :, :delta_ws[j]] = target_qs[j, :, -delta_ws[j]:]
                elif delta_ws[j] > 0:
                    target_qs[j, :, delta_ws[j]:] = target_qs[j, :, :-delta_ws[j]]
                # target_qs[j, :, :delta_ws[j]] = 0 # WARNING: this is only for forward moving windows like in Mario (can't go back)

        td_errors = self.train(obs, acs, target_qs, weights)
        return td_errors

    # reaching a position and dying is fine with this version
    def optimize(self, t):
        for iteration in range(self.optim_iters):
            samples = self.replay_buffer.sample_qmap(self.batch_size, t, self.n_steps)
            td_errors = self._optimize(*samples)
            self.replay_buffer.update_priorities_qmap(td_errors)

    def update_target(self):
        self.update_target()


class Q_Map_DQN_Agent:
    def __init__(self,
        # All
        observation_space,
        n_actions,
        coords_shape,
        double_replay_buffer,
        task_gamma,
        exploration_schedule,
        seed,
        path,
        learning_starts=1000,
        train_freq=1,
        print_freq=100,
        env_name='ENV',
        agent_name='AGENT',
        renderer_viewer=True,
        # DQN:
        dqn_q_func=None,
        dqn_lr=5e-4,
        dqn_batch_size=32,
        dqn_optim_iters=1,
        dqn_target_net_update_freq=500,
        dqn_grad_norm_clip=100,
        dqn_double_q=True,
        # Q-Map:
        q_map_model=None,
        q_map_random_schedule=None,
        q_map_greedy_bias=0.5,
        q_map_timer_bonus=0.5,
        q_map_lr=5e-4,
        q_map_gamma=0.9,
        q_map_n_steps=1,
        q_map_batch_size=32,
        q_map_optim_iters=1,
        q_map_target_net_update_freq=500,
        q_map_min_goal_steps=10,
        q_map_max_goal_steps=20,
        q_map_grad_norm_clip=1000,
        q_map_double_q=True
    ):

        # All

        self.observation_space = observation_space
        self.n_actions = n_actions
        self.coords_shape = coords_shape
        self.double_replay_buffer = double_replay_buffer
        self.task_gamma = task_gamma
        self.exploration_schedule = exploration_schedule
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.print_freq = print_freq

        agent_name += '-train' + str(train_freq)

        # DQN

        if dqn_q_func is not None:
            self.use_dqn = True
            agent_name += '-DQN-lr' + str(dqn_lr) + '-freq-' + str(train_freq)
            self.dqn_target_net_update_freq = dqn_target_net_update_freq

            self.dqn = DQN(
                model=dqn_q_func,
                observation_space=observation_space,
                n_actions=n_actions,
                gamma=task_gamma,
                lr=dqn_lr,
                replay_buffer=double_replay_buffer,
                batch_size=dqn_batch_size,
                optim_iters=dqn_optim_iters,
                grad_norm_clip=dqn_grad_norm_clip,
                double_q=dqn_double_q
            )
        else:
            self.use_dqn = False

        # Q-MAP

        if q_map_model is not None:
            agent_name += '-Q-MAP-' + q_map_model.description + '-' + str(q_map_min_goal_steps) + '-' + str(q_map_max_goal_steps) + '-gamma' + str(q_map_gamma) + '-lr' + str(q_map_lr) + '-bias' + str(q_map_greedy_bias) + '-bonus' + str(q_map_timer_bonus)
            self.use_q_map = True
            self.q_map_timer_bonus = q_map_timer_bonus
            self.using_q_map_starts = 2 * self.learning_starts
            self.q_map_random_schedule = q_map_random_schedule
            self.q_map_greedy_bias = q_map_greedy_bias
            self.q_map_goal_proba = 1 # TODO
            self.q_map_gamma = q_map_gamma
            self.q_map_target_net_update_freq = q_map_target_net_update_freq
            self.q_map_min_goal_steps = q_map_min_goal_steps
            self.q_map_max_goal_steps = q_map_max_goal_steps
            self.q_map_min_q_value = q_map_gamma ** (q_map_max_goal_steps-1)
            self.q_map_max_q_value = q_map_gamma ** (q_map_min_goal_steps-1)
            self.q_map_goal = None
            self.q_map_goal_timer = 0

            self.q_map = Q_Map(
                model=q_map_model,
                observation_space=observation_space,
                coords_shape=coords_shape,
                n_actions=n_actions,
                gamma=q_map_gamma,
                n_steps=q_map_n_steps,
                lr=q_map_lr,
                replay_buffer=double_replay_buffer,
                batch_size=q_map_batch_size,
                optim_iters=q_map_optim_iters,
                grad_norm_clip=q_map_grad_norm_clip,
                double_q=q_map_double_q
            )
        else:
            self.use_q_map = False

        if not self.use_dqn and not self.use_q_map:
            agent_name += '-random'

        else:
            self.tf_saver = tf.train.Saver()
            agent_name += '-memory' + str(double_replay_buffer._maxsize)

        # All

        sub_name = 'seed-{}_{}'.format(seed, datetime.utcnow().strftime('%F_%H-%M-%S-%f'))
        self.path = '{}/{}/{}/{}'.format(path, env_name, agent_name, sub_name)
        # log exploration for debugging
        exploration_labels = ['steps', 'planned exploration', 'current exploration', 'random actions', 'goal actions', 'greedy actions']
        self.exploration_logger = CSVLogger(exploration_labels, self.path + '/exploration')
        # videos etc.
        self.renderer = Q_Map_Renderer(self.path, viewer=renderer_viewer)
        # path to store
        self.tensorflow_path = self.path + '/tensorflow'
        if not os.path.exists(self.tensorflow_path):
            os.makedirs(self.tensorflow_path)

        U.initialize()
        self.t = 0
        self.episode_rewards = []
        self.random_proba = self.exploration_schedule.value(0)
        self.random_freq = self.exploration_schedule.value(0)
        self.greedy_freq = 1.0 - self.random_freq
        self.goal_freq = 0.0

        if self.use_dqn:
            self.dqn.update_target()

        self.seed(seed)

    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, ob):
        if self.use_q_map:
            self.q_map_goal_timer = 0
            self.q_map_goal = None

        frames = ob[0]
        ac = self.choose_action(ob)

        self.log()
        self.episode_rewards.append(0.0)
        self.prev_ob = ob
        self.prev_ac = ac

        return ac

    def step(self, ob, rew, done):
        prev_frames, (_, _, prev_w), _, _ = self.prev_ob
        frames, (row, col, w), _, _ = ob

        if self.double_replay_buffer is not None:
            self.double_replay_buffer.add(prev_frames, self.prev_ac, rew, (row, col-w, w-prev_w), frames, done)

        self.optimize()

        if not done:
            ac = self.choose_action(ob)
        else:
            ac = None
            self.add_to_renderer(ob)

        self.t += 1
        self.episode_rewards[-1] += rew
        self.prev_ob = ob
        self.prev_ac = ac

        return ac

    def choose_action(self, ob):
        frames, (row, col, w), screen, (full_r, full_c) = ob

        q_map_values = None
        q_map_candidates = []
        q_map_biased_candidates = []

        # render Q-maps all the time even if we do not need them
        if self.use_q_map:
            q_map_values = self.q_map.compute_q_values(frames[None])[0] # (rows, cols, acs)

        if self.np_random.rand() < self.random_proba or (not self.use_dqn and self.t <= self.using_q_map_starts):
            ac = self.np_random.randint(self.n_actions)
            action_type = 'random'

        else:
            # Q-Map available and started to train
            if self.use_q_map and self.t > self.using_q_map_starts:
                # reached goal
                if self.q_map_goal_timer > 0 and self.q_map_goal[1] < w:
                    self.q_map_goal_timer = 0
                    self.q_map_goal = None

                # goal unreachable
                if self.q_map_goal_timer > 0 and (row, col) == self.q_map_goal:
                    self.q_map_goal_timer = 0
                    self.q_map_goal = None

                # no more goal
                if self.q_map_goal_timer == 0:
                    if self.np_random.rand() < self.q_map_goal_proba:
                        # find a new goal
                        q_map_max_values = q_map_values.max(2) # (rows, cols)
                        q_map_candidates_mask = np.logical_and(self.q_map_min_q_value <= q_map_max_values, self.q_map_max_q_value >= q_map_max_values)
                        q_map_candidates = np.where(q_map_candidates_mask)
                        q_map_candidates = np.dstack(q_map_candidates)[0] # list of (row, col)

                        if len(q_map_candidates) > 0:
                            # goals compatible with greedy action
                            if self.use_dqn and self.np_random.rand() < self.q_map_greedy_bias:
                                greedy_ac = self.dqn.choose_action(frames, stochastic=False)
                                q_map_biased_candidates_mask = np.logical_and(q_map_candidates_mask, q_map_values.argmax(2) == greedy_ac)
                                q_map_biased_candidates = np.where(q_map_biased_candidates_mask)
                                q_map_biased_candidates = np.dstack(q_map_biased_candidates)[0] # list of (row, col)

                            # same DQN and Q-Map action
                            if len(q_map_biased_candidates) > 0:
                                goal_idx = self.np_random.randint(len(q_map_biased_candidates))
                                q_map_goal_row, q_map_goal_col_local = q_map_biased_candidates[goal_idx]
                                q_map_expected_steps = math.log(q_map_max_values[q_map_goal_row, q_map_goal_col_local], self.q_map_gamma) + 1
                                self.q_map_goal_timer = math.ceil(1.5 * q_map_expected_steps) # 50% bonus
                                self.q_map_goal = (q_map_goal_row, q_map_goal_col_local + w)
                                ac = greedy_ac
                                action_type = 'dqn/qmap'

                            # greedy Q-Map action
                            else:
                                goal_idx = self.np_random.randint(len(q_map_candidates))
                                q_map_goal_row, q_map_goal_col_local = q_map_candidates[goal_idx]
                                q_map_expected_steps = math.log(q_map_max_values[q_map_goal_row, q_map_goal_col_local], self.q_map_gamma) + 1
                                self.q_map_goal_timer = math.ceil((1. + self.q_map_timer_bonus) * q_map_expected_steps)
                                self.q_map_goal = (q_map_goal_row, q_map_goal_col_local + w)
                                ac, q_map_values = self.q_map.choose_action(None, (q_map_goal_row, q_map_goal_col_local), q_map_values) # no need to recompute the Q-Map
                                action_type = 'qmap'

                            self.q_map_goal_timer -= 1
                            if self.q_map_goal_timer == 0:
                                self.q_map_goal = None

                        # random action
                        else:
                            self.q_map_goal_timer = 0
                            self.q_map_goal = None
                            ac = self.np_random.randint(self.n_actions)
                            action_type = 'random'

                    # DQN action
                    else:
                        ac = self.dqn.choose_action(frames, stochastic=False)
                        action_type = 'dqn'

                # Q-Map action
                else:
                    q_map_goal_row, q_map_goal_col = self.q_map_goal
                    q_map_goal_col_local = q_map_goal_col - w
                    ac, q_map_values = self.q_map.choose_action(frames, (q_map_goal_row, q_map_goal_col_local))
                    self.q_map_goal_timer -= 1
                    if self.q_map_goal_timer == 0:
                        self.q_map_goal = None
                    action_type = 'qmap'

            # DQN action
            else:
                ac = self.dqn.choose_action(frames, stochastic=False)
                action_type = 'dqn'

        # rendering
        self.add_to_renderer(ob, q_map_values, ac, action_type, q_map_candidates, q_map_biased_candidates)

        # update exploration
        if action_type == 'dqn/qmap':
            self.random_freq += 0.01 * (0 - self.random_freq)
            self.greedy_freq += 0.01 * (1 - self.greedy_freq)
            self.goal_freq += 0.01 * (0 - self.goal_freq) # TODO: 1?
        elif action_type == 'dqn':
            self.random_freq += 0.01 * (0 - self.random_freq)
            self.greedy_freq += 0.01 * (1 - self.greedy_freq)
            self.goal_freq += 0.01 * (0 - self.goal_freq)
        elif action_type == 'qmap':
            self.random_freq += 0.01 * (0 - self.random_freq)
            self.greedy_freq += 0.01 * (0 - self.greedy_freq)
            self.goal_freq += 0.01 * (1 - self.goal_freq)
        elif action_type == 'random':
            self.random_freq += 0.01 * (1 - self.random_freq)
            self.greedy_freq += 0.01 * (0 - self.greedy_freq)
            self.goal_freq += 0.01 * (0 - self.goal_freq)
        else:
            raise NotImplementedError('unknown action type {}'.format(action_type))

        target_exploration = self.exploration_schedule.value(self.t)
        current_exploration = (1.0 - self.greedy_freq)
        if self.use_q_map and self.t >= self.using_q_map_starts:
            self.random_proba = self.q_map_random_schedule.value(self.t)
            if current_exploration > target_exploration:
                self.q_map_goal_proba -= 0.001
            elif current_exploration < target_exploration:
                self.q_map_goal_proba += 0.001
        else:
            self.random_proba = self.exploration_schedule.value(self.t)

        if (self.t+1) % 100 == 0:
            self.exploration_logger.log(self.t+1, target_exploration, current_exploration, self.random_freq, self.goal_freq, self.greedy_freq)

        return ac

    def optimize(self):
        if (self.use_dqn or self.use_q_map) and self.t >= self.learning_starts and self.t % self.train_freq == 0:
            if self.use_dqn:
                self.dqn.optimize(self.t)

            if self.use_q_map:
                self.q_map.optimize(self.t)

        if self.use_dqn and self.t >= self.learning_starts and self.t % self.dqn_target_net_update_freq == 0:
            self.dqn.update_target()

        if self.use_q_map and self.t >= self.learning_starts and self.t % self.q_map_target_net_update_freq == 0:
            self.q_map.update_target()

        # save the session
        if (self.use_dqn or self.use_q_map) and (self.t+1) % 100000 == 0:
            file_name = self.tensorflow_path  + '/step_' + str(self.t+1) + '.ckpt'
            print('saving tensorflow session to', file_name)
            self.tf_saver.save(tf.get_default_session(), file_name)

    def log(self):
        if self.t > 0 and self.print_freq is not None and len(self.episode_rewards) % self.print_freq == 0:
            mean_100ep_reward = np.mean(self.episode_rewards[-100:])
            num_episodes = len(self.episode_rewards)

            logger.record_tabular('steps', self.t)
            logger.record_tabular('episodes', num_episodes)
            logger.record_tabular('mean 100 episode reward', '{:.3f}'.format(mean_100ep_reward))
            logger.record_tabular('exploration (target)', '{:.3f} %'.format(100 * self.exploration_schedule.value(self.t)))
            logger.record_tabular('exploration (current)', '{:.3f} %'.format(100 * (1.0 - self.greedy_freq)))
            logger.dump_tabular()

    def load(self, path):
        cwd = os.getcwd()
        path, file = os.path.split(path)
        os.chdir(path)
        self.tf_saver.restore(tf.get_default_session(), file)
        os.chdir(cwd)
        print('model restored :)')

    def add_to_renderer(self, ob, q_map_values=None, ac=None, action_type='', q_map_candidates=[], q_map_biased_candidates=[]):
        if self.renderer is not None:
            if self.use_q_map and self.q_map_goal is not None:
                goal = self.q_map_goal
                assert self.q_map_goal_timer > 0
            else:
                goal = None
            self.renderer.add(ob, self.coords_shape, q_map_values, ac, action_type, self.n_actions, q_map_candidates, q_map_biased_candidates, goal)
