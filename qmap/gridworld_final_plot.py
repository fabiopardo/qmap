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

def smooth(data, window, fast):
    labels = list(data.keys())
    steps = np.concatenate(data['steps'])
    idx = np.argsort(steps)
    loss = np.concatenate(data['loss'])
    loss_queue = deque(maxlen=window)
    means = []
    for i in idx:
        loss_queue.append(loss[i])
        if len(loss_queue) < window: continue
        means.append(np.mean(loss_queue))
        if fast:
            loss_queue.clear()
    data['loss'] = means
    x_queue = deque(maxlen=window)
    means = []
    for i in idx:
        x_queue.append(steps[i])
        if len(x_queue) < window: continue
        means.append(np.mean(x_queue))
        if fast:
            x_queue.clear()
    data['steps'] = np.array(means)
    data['steps'] *= 32 # WARNING: THIS IS ONLY FOR BATCH OF 32
    data['steps'] /= 1e6
    data['loss'] = np.array(data['loss'])
    data['loss'] /= 1e-3


def plot(path, window, fast, room):
    room_name = 'level{}'.format(room)
    colors = [plt.get_cmap('tab10').colors[2], plt.get_cmap('tab10').colors[0]]
    fig, ax = plt.subplots(figsize=(3, 2))
    # ax.set_xlabel('transitions (millions)')
    # ax.set_ylabel('ground truth MSE ($\\times 10^{-3}$)')
    ax.set_axisbelow(True)
    ax.grid(color='#dddddd')
    ax.set_axisbelow(True)

    print('Searching for logs in:', path)
    os.chdir(path)
    os.chdir('GridWorld_window32_scale1-v0')
    for agent_id in filter(os.path.isdir, sorted(os.listdir('.'))):
        if agent_id == 'Qmap_convdeconv1_gamma0.9_dueling_double_layernorm_lr0.0001_batch32_target1000':
            agent_name = 'Conv'
            agent_idx = 0
        elif agent_id == 'Qmap_mlp1_gamma0.9_dueling_double_layernorm_lr0.0001_batch32_target1000':
            agent_name = 'MLP'
            agent_idx = 1
        else:
            print('ignore agent', agent_id)
            continue
        data = {'steps': [], 'loss': []}
        os.chdir(agent_id)
        print(' '*2, agent_id, agent_name)
        for run_id in filter(os.path.isdir, sorted(os.listdir('.'))):
            os.chdir(run_id)
            print(' '*4, run_id)
            csv = pd.read_csv('ground_truth_loss.csv')
            steps = csv['steps']
            data['steps'].append(steps)
            room_data = csv[room_name]
            data['loss'].append(room_data)
            os.chdir('..')
        os.chdir('..')
        smooth(data, window, fast)
        ax.plot(data['steps'], data['loss'], color=colors[agent_idx], alpha=1, lw=1, label=agent_name)
    os.chdir('..')

    ax.set_ylim(0, 5)
    ax.set_xlim(0, 50)
    if room == 1:
        ax.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.canvas.draw()
    figure_name = 'gridworld_loss_{}.pdf'.format(room_name)
    fig.savefig(figure_name)
    print('Folder:', os.getcwd())
    print('Figure saved in:', figure_name)

if __name__ == '__main__':
    import argparse
    from baselines.common.misc_util import boolean_flag

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', required=True)
    parser.add_argument('--window', type=int, default=int(1000))
    parser.add_argument('--room', type=int, default=1)
    boolean_flag(parser, 'fast', default=True)
    args = parser.parse_args()
    plot(**vars(args))
