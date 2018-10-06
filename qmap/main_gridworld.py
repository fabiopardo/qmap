import argparse
from baselines.common import set_global_seeds
from baselines.common.misc_util import boolean_flag
from baselines.common.schedules import LinearSchedule
import baselines.common.tf_util as U
from baselines import deepq
from csv_logger import CSVLogger
from gym.envs.classic_control import rendering
from gym.utils import seeding
from gym.wrappers import TimeLimit
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.misc import toimage
import tensorflow as tf
from termcolor import colored
import time

from dqn_agent import DQN_Agent
from envs.gridworld import GridWorld
from envs.wrappers import PerfLogger
from models import ConvDeconvMap, MlpMap
from q_map_dqn_agent import Q_Map


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--model', default='convdeconv1')
parser.add_argument('--target', type=int, default=1000)
parser.add_argument('--path', required=True)
boolean_flag(parser, 'dueling', default=True)
boolean_flag(parser, 'norm', default=True)
boolean_flag(parser, 'double', default=True)
boolean_flag(parser, 'render', default=False)
args = parser.parse_args()

n_steps = int(1e8)

train_level = 'level1'
test_levels = ['level1', 'level2', 'level3']

env = GridWorld(train_level)
coords_shape = env.unwrapped.coords_shape
set_global_seeds(args.seed)
env.seed(args.seed)

print('~~~~~~~~~~~~~~~~~~~~~~')
print(env.spec.id)
print('observations:', env.observation_space.shape)
print('coords:     ', coords_shape)
print('actions:    ', env.action_space.n)
print('walls:      ', env.unwrapped.walls.shape)
print('~~~~~~~~~~~~~~~~~~~~~~')

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.__enter__()

if args.model == 'convdeconv1':
    q_map_model = ConvDeconvMap(
        convs=[(32, 8, 2), (32, 6, 2), (64, 4, 1)],
        middle_hiddens=[1024],
        deconvs=[(64, 4, 1), (32, 6, 2), (env.action_space.n, 4, 2)],
        coords_shape=coords_shape,
        dueling=args.dueling,
        layer_norm=args.norm,
        activation_fn=tf.nn.elu
    )
elif args.model == 'convdeconv2':
    q_map_model = ConvDeconvMap(
        convs=[(32, 8, 2), (32, 6, 2), (64, 4, 1)],
        middle_hiddens=[1024],
        deconvs=[(64, 4, 1), (32, 6, 2), (env.action_space.n, 8, 2)],
        coords_shape=coords_shape,
        dueling=args.dueling,
        layer_norm=args.norm,
        activation_fn=tf.nn.elu
    )
elif args.model == 'convdeconv3':
    q_map_model = ConvDeconvMap(
        convs=[(32, 4, 2), (32, 4, 2), (32, 4, 1)],
        middle_hiddens=[512],
        deconvs=[(32, 4, 1), (32, 4, 2), (env.action_space.n, 4, 2)],
        coords_shape=coords_shape,
        dueling=args.dueling,
        layer_norm=args.norm,
        activation_fn=tf.nn.elu
    )
elif args.model == 'mlp1':
    q_map_model = MlpMap(
        hiddens=[1024, 1024, 1024],
        coords_shape=coords_shape,
        dueling=args.dueling,
        layer_norm=args.norm,
        activation_fn=tf.nn.elu
    )
elif args.model == 'mlp2':
    q_map_model = MlpMap(
        hiddens=[1024, 1024],
        coords_shape=coords_shape,
        dueling=args.dueling,
        layer_norm=args.norm,
        activation_fn=tf.nn.elu
    )
q_map = Q_Map(
    model=q_map_model,
    observation_space=env.observation_space,
    coords_shape=env.unwrapped.coords_shape,
    n_actions=env.action_space.n,
    gamma=args.gamma,
    n_steps=1,
    lr=args.lr,
    replay_buffer=None,
    batch_size=args.batch,
    optim_iters=1,
    grad_norm_clip=1000,
    double_q=args.double
)
U.initialize()

plt.ion()
test_obs = []
test_qmaps = []
image_indexes = []
n_images = 20
for level in test_levels:
    test_obs.append(np.load('gridworld_obs_{}.npy'.format(level)))
    test_qmaps.append(np.load('gridworld_gound_truth_{}.npy'.format(level)))
    image_indexes.append(np.linspace(300, len(test_obs[-1])-300, n_images).astype(int))
if args.render:
    viewer = rendering.SimpleImageViewer(maxwidth=2500)
agent_name = 'Qmap_{}_gamma{}_{}{}{}lr{}_batch{}_target{}'.format(args.model, args.gamma, 'dueling_' if args.dueling else '', 'double_' if args.double else '', 'layernorm_' if args.norm else '', args.lr, args.batch, args.target)
path = '{}/{}/{}'.format(args.path, agent_name, args.seed)
loss_logger = CSVLogger(['steps'] + test_levels, os.path.expanduser('{}/ground_truth_loss'.format(path)))
os.mkdir('{}/images'.format(path))
color_map = plt.get_cmap('inferno')


batch_weights = np.ones(q_map.batch_size)
batch_dones = np.zeros((q_map.batch_size, 1))
for t in range(n_steps // q_map.batch_size + 1):
    batch_prev_frames = []
    batch_ac = []
    batch_rcw = []
    batch_frames = []
    for _ in range(q_map.batch_size):
        prev_ob = env.random_reset()
        ac = env.action_space.sample()
        ob = env.step(ac)[0]
        prev_frames, (_, _, prev_w), _, _ = prev_ob
        frames, (row, col, w), _, _ = ob
        batch_prev_frames.append(prev_frames)
        batch_ac.append(ac)
        batch_rcw.append((row, col-w, w-prev_w))
        batch_frames.append(frames)
    batch_prev_frames = np.array(batch_prev_frames)
    batch_ac = np.array(batch_ac)
    batch_rcw = np.array(batch_rcw)[:, None, :]
    batch_frames = np.array(batch_frames)
    q_map._optimize(batch_prev_frames, batch_ac, batch_rcw, batch_frames, batch_dones, batch_weights)
    if t % args.target == 0:
        q_map.update_target()

    if t % 50 == 0:
        losses = []
        all_images = []
        for i_level in range(len(test_levels)):
            pred_qmaps = q_map.compute_q_values(test_obs[i_level])
            true_qmaps = test_qmaps[i_level]
            loss = np.mean((pred_qmaps - true_qmaps)**2)
            losses.append(loss)
            ob_images = np.concatenate(test_obs[i_level][image_indexes[i_level]], axis=1)
            pred_images = np.concatenate((color_map(pred_qmaps[image_indexes[i_level]].max(3))[:, :, :, :3] * 255).astype(np.uint8), axis=1)
            true_images = np.concatenate((color_map(true_qmaps[image_indexes[i_level]].max(3))[:, :, :, :3] * 255).astype(np.uint8), axis=1)
            all_images.append(np.concatenate((ob_images, true_images, pred_images), axis=0))
        img = np.concatenate(all_images, axis=0)
        toimage(img, cmin=0, cmax=255).save('{}/images/{}.png'.format(path, t))
        if args.render:
            img = np.repeat(np.repeat(img, 3, 0), 3, 1)
            viewer.imshow(img)
        print(t*args.batch, 'Losses:', *losses)
        loss_logger.log(t, *losses)
