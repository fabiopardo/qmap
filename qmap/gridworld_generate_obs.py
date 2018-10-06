import argparse
from qmap.envs.gridworld import GridWorld
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--level', default='level1')
args = parser.parse_args()

env = GridWorld(args.level)
all_coords = []
all_obs = []
render = False
for c in range(1, env.cols-1):
    for r in range(1, env.rows-1):
        if env.walls[r, c]: continue
        w = c - env.padding
        all_coords.append((r, c, w))
        env.r = r
        env.c = c
        env.w = w
        all_obs.append(env.get_obs()[0])
        if render: env.render()
all_coords = np.array(all_coords)
all_obs = np.array(all_obs)
np.save('gridworld_coords_{}'.format(args.level), all_coords)
np.save('gridworld_obs_{}'.format(args.level), all_obs)
