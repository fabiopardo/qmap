import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque
import numpy as np
import os
import pandas as pd
import scipy as sp

plt.rcParams.update({'font.size': 12})
plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"

def smooth(data, window, freq):
    labels = list(data.keys())
    steps = np.concatenate(data['steps'])
    idx = np.argsort(steps)
    scores = np.concatenate(data['scores'])
    scores_queue = deque(maxlen=window)
    means = []
    stds = []
    mins = []
    maxs = []
    for i in idx:
        scores_queue.append(scores[i])
        if len(scores_queue) < window or i%freq != 0: continue
        means.append(np.mean(scores_queue))
        # stds.append(np.std(x_queue))
        confidence = 0.99
        se = sp.stats.sem(scores_queue)
        h = se * sp.stats.t.ppf((1+confidence)/2., len(scores_queue)-1)
        stds.append(h)
        mins.append(np.min(scores_queue))
        maxs.append(np.max(scores_queue))
    data['scores'] = means
    data['scores std'] = np.array(stds)
    data['scores min'] = np.array(mins)
    data['scores max'] = np.array(maxs)
    x_queue = deque(maxlen=window)
    means = []
    for i in idx:
        x_queue.append(steps[i])
        if len(x_queue) < window or i%freq != 0: continue
        means.append(np.mean(x_queue))
    data['steps'] = np.array(means)
    data['steps'] /= 1e6


def plot(path, window, freq):
    colors = plt.get_cmap('tab10').colors[2:4]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.set_xlabel('timesteps (millions)')
    ax.set_ylabel('sum of rewards')
    ax.set_axisbelow(True)
    ax.grid(color='#dddddd')
    ax.set_axisbelow(True)

    print('Searching for logs in:', path)
    os.chdir(path)
    os.chdir('CustomSuperMarioAllStars_obs56x64xGx3_skip2_repeat4-v0')
    for agent_id in filter(os.path.isdir, sorted(os.listdir('.'))):
        if agent_id == '5M-train4-DQN-lr0.0001-freq-4-Q-MAP-ConvDeconvMap-[(32,8,2),(32,6,2),(64,4,2),1024,(64,4,2),(32,6,2),(6,4,1)]-elu-duel-norm-15-30-gamma0.9-lr0.0003-bias0.5-bonus0.5-memory500000':
            agent_name = 'DQN + Q-map'
            agent_idx = 0
        elif agent_id == '5M-train4-DQN-lr0.0001-freq-4-memory500000':
            agent_name = 'DQN'
            agent_idx = 1
        else:
            print('ignore agent', agent_id)
            continue
        data = {'steps': [], 'scores': []}
        flags = []
        os.chdir(agent_id)
        print(' '*2, agent_id, agent_name)
        for run_id in filter(os.path.isdir, sorted(os.listdir('.'))):
            os.chdir(run_id)
            print(' '*4, run_id)
            csv = pd.read_csv('score.csv')
            steps = csv['steps']
            data['steps'].append(steps)
            scores = csv['undiscounted return']
            data['scores'].append(scores)
            for file in os.listdir('coords'):
                if 'distance_3161' in file:
                    split = file.split('_')
                    assert split[-2] == 'episode'
                    episode = int(split[-1][:-4]) - 1
                    step = steps[episode] / 1e6
                    score = scores[episode]
                    print('flag on step', step, 'with score', score)
                    flags.append(step)
            os.chdir('..')
        os.chdir('..')
        smooth(data, window, freq)
        ax.fill_between(data['steps'], data['scores']-data['scores std'], data['scores']+data['scores std'], color=colors[agent_idx], alpha=0.15, lw=0)
        ax.plot(data['steps'], data['scores'], color=colors[agent_idx], alpha=1, lw=1, label=agent_name)
        # find the closest step on the curve
        for step in flags:
            closest_idx = (np.abs(data['steps'] - step)).argmin()
            closest_step = data['steps'][closest_idx]
            closest_score = data['scores'][closest_idx]
            print('looking for', step, 'found', closest_step)
            ax.plot(closest_step, closest_score, '|', alpha=0.8, markersize=10, color='k')
            ax.plot(closest_step, closest_score, '|', alpha=0.8, markersize=7, color=colors[agent_idx])
    os.chdir('..')

    ax.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.canvas.draw()
    figure_name = 'mario_score_and_flags.pdf'
    fig.savefig(figure_name)
    print('Folder:', os.getcwd())
    print('Figure saved in:', figure_name)

if __name__ == '__main__':
    import argparse
    from baselines.common.misc_util import boolean_flag

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', required=True)
    parser.add_argument('--window', type=int, default=int(10))
    parser.add_argument('--freq', type=int, default=1)
    args = parser.parse_args()
    plot(**vars(args))
