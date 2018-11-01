from collections import deque, OrderedDict
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import pandas as pd


plt.rcParams.update({'font.size': 4})


def smooth(data, window, freq):
    labels = list(data.keys())
    steps = np.concatenate(data['steps'])
    idx = np.argsort(steps)
    for l in labels:
        if l == 'steps': continue
        x = np.concatenate(data[l])
        x_queue = deque(maxlen=window)
        means = []
        stds = []
        mins = []
        maxs = []
        for i in idx:
            x_queue.append(x[i])
            if len(x_queue) < window or i%freq != 0: continue
            means.append(np.mean(x_queue))
            stds.append(np.std(x_queue))
            mins.append(np.min(x_queue))
            maxs.append(np.max(x_queue))
        data[l] = means
        data[l+' std'] = np.array(stds)
        data[l+' min'] = np.array(mins)
        data[l+' max'] = np.array(maxs)
    x = steps
    x_queue = deque(maxlen=window)
    means = []
    for i in idx:
        x_queue.append(x[i])
        if len(x_queue) < window or i%freq != 0: continue
        means.append(np.mean(x_queue))
    data['steps'] = np.array(means)


def plot(path, window, max_step, mini, maxi, freq, without, only, averaged, std, extrema, logscale):
    print('Searching for logs in:', path)
    os.chdir(path)
    for env_id in filter(os.path.isdir, sorted(os.listdir('.'))):
        os.chdir(env_id)
        print(env_id)
        data = {}
        labels = {}
        for agent_id in filter(os.path.isdir, sorted(os.listdir('.'))):
            os.chdir(agent_id)
            print(' '*2, agent_id)
            for run_id in filter(os.path.isdir, sorted(os.listdir('.'))):
                if without is not None and without in env_id+'/'+agent_id+'/'+run_id: continue
                if only is not None and not only in env_id+'/'+agent_id+'/'+run_id: continue
                os.chdir(run_id)
                print(' '*4, run_id)
                for file in os.listdir('.'):
                    if file.endswith('.csv'):
                        log = file[:-4]
                        csv = pd.read_csv(log + '.csv')
                        if not log in labels:
                            labels[log] = list(csv.axes[1])
                        else:
                            assert labels[log] == list(csv.axes[1])
                        assert 'steps' in labels[log]
                        run_label = agent_id
                        if not averaged: run_label += '_' + run_id
                        if not log in data:
                            data[log] = OrderedDict()
                        if not run_label in data[log]:
                            data[log][run_label] = {l: [] for l in labels[log]}
                        for l in labels[log]:
                            data[log][run_label][l] += [csv[l]]
                os.chdir('..')
            os.chdir('..')
        os.chdir('..')
        for log in data:
            n = len(data[log])
            if n != 0:
                if n <= 10:
                    colors = plt.get_cmap('tab10').colors
                elif n <= 20:
                    colors = plt.get_cmap('tab20').colors
                elif n <= 256:
                    print(np.rint(np.linspace(0, 255, n)).astype(int))
                    colors = plt.get_cmap('gist_rainbow')
                    colors = [colors(i) for i in np.rint(np.linspace(0, 255, n)).astype(int)]
                else:
                    raise KeyError('Too many lines to plot! {}'.format(n))
                figure_name = env_id
                if not averaged:
                    figure_name += '_no-averaged'
                fig = plt.figure(figsize=(6, 6.5))
                fig.suptitle(figure_name)
                labels[log].remove('steps')
                grid = gridspec.GridSpec(2*len(labels[log])+2, 1)
                axes = []
                for i, l in enumerate(labels[log]):
                    ax = fig.add_subplot(grid[2*i:2*i+2, 0])
                    ax.set_xlabel('steps')
                    ax.set_ylabel(l)
                    ax.grid(color='#dddddd')
                    ax.set_axisbelow(True)
                    if logscale:
                        ax.set_yscale('log', nonposy='clip')
                    axes.append(ax)
                legend_ax = fig.add_subplot(grid[-2:, 0])
                legend_ax.axis('off')
                for i, run_label in enumerate(data[log]):
                    run_data = data[log][run_label]
                    n = len(run_data['steps'])
                    if n != 1:
                        run_label += ' ({})'.format(n)
                    smooth(run_data, window, freq)
                    color = colors[i]
                    if std:
                        for i, l in enumerate(labels[log]):
                            axes[i].fill_between(run_data['steps'], run_data[l]-run_data[l+' std'],
                                                 run_data[l]+run_data[l+' std'],
                                                 color=color, alpha=0.15, lw=0)
                    if extrema:
                        for i, l in enumerate(labels[log]):
                            axes[i].plot(run_data['steps'], run_data[l+' min'], color=color, alpha=0.5, lw=0.5)
                            axes[i].plot(run_data['steps'], run_data[l+' max'], color=color, alpha=0.5, lw=0.5)
                    for i, l in enumerate(labels[log]):
                        axes[i].plot(run_data['steps'], run_data[l], color=color, alpha=0.5, lw=1, label=run_label)
                        if mini is not None and maxi is not None:
                            axes[i].set_ylim(mini, maxi)
                handles, labels[log] = axes[0].get_legend_handles_labels()
                legend_ax.legend(handles, labels[log], loc='center')
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                fig.canvas.draw()
                figure_name += '_' + log + '.pdf'
                fig.savefig(figure_name)
                print('Folder:', os.getcwd())
                print('Figure saved in:', figure_name)


if __name__ == '__main__':
    import argparse
    from baselines.common.misc_util import boolean_flag

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', default='qmap_results')
    parser.add_argument('--window', type=int, default=int(10))
    parser.add_argument('--max-step', type=int, default=None)
    parser.add_argument('--mini', type=float, default=None)
    parser.add_argument('--maxi', type=float, default=None)
    parser.add_argument('--without', default=None)
    parser.add_argument('--only', default=None)
    parser.add_argument('--freq', type=int, default=1)
    boolean_flag(parser, 'averaged', default=True)
    boolean_flag(parser, 'std', default=True)
    boolean_flag(parser, 'extrema', default=True)
    boolean_flag(parser, 'logscale', default=False)
    args = parser.parse_args()
    plot(**vars(args))
