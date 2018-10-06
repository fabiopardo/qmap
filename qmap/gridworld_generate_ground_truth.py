import argparse
from qmap.envs.gridworld import GridWorld
import numpy as np
import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--level', default='level1')
args = parser.parse_args()

np.set_printoptions(precision=3, linewidth=300, edgeitems=100)
plt.ion()
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
for ax in [axes[0, 0], axes[0, 2], axes[2, 0], axes[2, 2]]:
    ax.axis('off')
ac_axes = [axes[0, 1], axes[1, 0], axes[1, 2], axes[2, 1]]
max_ax = axes[1,1]

def ground_truth_distances(env, w):
    walls = env.padded_walls[:,env.padding+w:env.padding+w+env.window_size]
    x = np.full((env.rows, env.window_size+2), np.inf)
    x[env.r, env.c-w+1] = 0

    while True:
        next_x = x.copy()
        next_x[1:-1,1:-1] = np.minimum.reduce([x[1:-1,1:-1], 1+x[0:-2,1:-1], 1+x[2:,1:-1], 1+x[1:-1,0:-2], 1+x[1:-1,2:]])
        next_x[:, 1:-1][walls] = np.inf
        if np.all(next_x == x):
            break
        x = next_x
    x = np.power(0.9, x[:,1:-1])
    return x

env = GridWorld(args.level)
path = 'gridworld_coords_{}.npy'.format(args.level)
print('loading coordinates from {}...'.format(path))
all_coords = np.load(path)
n = len(all_coords)
print('{} coordinates found'.format(n))
all_ground_truth = []
indexes = {}
print('generating ground truth q-maps...')
n_prints = 100
for i, (r, c, w) in enumerate(all_coords):
    actions_ground_truth = []
    for a in range(4):
        env.r = r
        env.c = c
        env.w = w
        env.step(a)
        ground_truth = ground_truth_distances(env, w)
        # take the movement of the window into account
        dw = env.w-w
        # if dw != 0:
        #     ground_truth[:, dw:] = ground_truth[:, :env.window_size-dw]
        #     ground_truth[:, :dw] = 0
        actions_ground_truth.append(ground_truth)
    all_ground_truth.append(actions_ground_truth)

    if (i+1)%(n//n_prints) == 0:
        env.render()
        print('{}%'.format(round(100*(i+1)/n)))
        for a in range(4):
            ac_axes[a].clear()
            ac_axes[a].imshow(actions_ground_truth[a], 'inferno')
        max_ax.clear()
        max_ax.imshow(np.stack(actions_ground_truth, axis=2).max(2), 'inferno')
        plt.pause(1e-10)
all_ground_truth = np.array(all_ground_truth)
all_ground_truth = np.moveaxis(all_ground_truth, 1, -1)
np.save('gridworld_gound_truth_{}'.format(args.level), all_ground_truth)
plt.waitforbuttonpress()
