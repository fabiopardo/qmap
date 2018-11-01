from gym import Env, spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
import numpy as np
import os


class GridWorld(Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, level='level1', scale=1):
        self.level = level
        if not '.' in level: level += '.bmp'
        self.walls = np.logical_not(plt.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), level)))
        self.height = self.walls.shape[0]
        self.width = 32
        # observations
        self.screen_shape = (self.height, self.width)
        self.padding = self.width // 2 - 1
        self.padded_walls = np.logical_not(np.pad(np.logical_not(self.walls), ((0, 0), (self.padding, self.padding)), 'constant'))
        self.observation_space = spaces.Box(0, 255, (self.height, self.width, 3), dtype=np.float32)
        # coordinates
        self.scale = scale
        self.coords_shape = (self.height // scale, self.width // scale)
        self.available_coords = np.array(np.where(np.logical_not(self.walls))).transpose()
        # actions
        self.action_space = spaces.Discrete(4)
        # miscellaneous
        self.name = 'GridWorld_obs{}x{}x3_qframes{}x{}x4-v0'.format(*self.screen_shape, *self.coords_shape)
        self.viewer = None
        self.seed()

    def next(self, r, c, ac):
        if   ac == 0: r1, c1 = r-1, c # UP
        elif ac == 1: r1, c1 = r, c-1 # LEFT
        elif ac == 2: r1, c1 = r, c+1 # RIGHT
        elif ac == 3: r1, c1 = r+1, c # DOWN
        else: raise KeyError('invalid action ' + str(ac))
        if not self.walls[r1, c1]:
            self.r, self.c, self.w = r1, c1, c1 - self.padding

    def random_reset(self):
        self.r, self.c = self.available_coords[self.np_random.randint(len(self.available_coords))]
        self.w = self.c - self.padding
        frames, rcw, full_rc = self.get_obs()
        return frames, rcw, frames, full_rc

    def step(self, action):
        self.next(self.r, self.c, action)
        frames, rcw, full_rc = self.get_obs()
        rew = 0
        done = False
        ob =  frames, rcw, frames, full_rc
        return ob, rew, done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_obs(self):
        frames = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frames[:, :, 2] = self.padded_walls[:, self.padding+self.w:self.padding+self.w+self.width] * 255
        frames[self.r, self.padding, :] = [255, 255, 0]
        r, c, w = self.r // self.scale, self.c // self.scale, self.w // self.scale
        assert r < self.coords_shape[0] and c-w < self.coords_shape[1], ((r, c, w), (self.r, self.c, self.w), self.coords_shape)
        self.full_c = self.c
        return frames, (r, c, w), (self.r, self.c)

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
        img = self.get_obs()[0]
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            from scipy.ndimage import zoom
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            img = zoom(img, [5, 5, 1], order=0)
            self.viewer.imshow(img)
        else:
            raise NotImplementedError

    # Generate ground truth Q-frames by finding the smallest number of steps towards all coordinates given a window position.
    def ground_truth_distances(self, w):
        walls = self.padded_walls[:, self.padding+w:self.padding+w+self.width]
        x = np.full((self.height, self.width + 2), np.inf)
        x[self.r, self.c-w+1] = 0
        while True:
            next_x = x.copy()
            next_x[1:-1,1:-1] = np.minimum.reduce([x[1:-1,1:-1], 1+x[0:-2,1:-1], 1+x[2:,1:-1], 1+x[1:-1,0:-2], 1+x[1:-1,2:]])
            next_x[:, 1:-1][walls] = np.inf
            if np.all(next_x == x):
                break
            x = next_x
        x = np.power(0.9, x[:,1:-1])
        return x

    def generate_ground_truth_qframes(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        print('Generating possible observations and coordinates...')

        all_coords = []
        all_obs = []
        render = False
        for c in range(1, self.width - 1):
            for r in range(1, self.height - 1):
                if self.walls[r, c]: continue
                w = c - self.padding
                all_coords.append((r, c, w))
                self.r = r
                self.c = c
                self.w = w
                all_obs.append(self.get_obs()[0])
                if render: self.render()
        all_coords = np.array(all_coords)
        all_obs = np.array(all_obs)
        obs_path = '{}/gridworld_obs_{}'.format(path, self.level)
        np.save(obs_path, all_obs)
        n = len(all_coords)
        print('{} coordinates found'.format(n))
        print('Coordiantes saved in {}'.format(obs_path))

        print('Generating ground truth Q-frames...')

        np.set_printoptions(precision=3, linewidth=300, edgeitems=100)
        plt.ion()
        fig, axes = plt.subplots(3, 3, figsize=(8, 8))
        for ax in [axes[0, 0], axes[0, 2], axes[2, 0], axes[2, 2]]:
            ax.axis('off')
        ac_axes = [axes[0, 1], axes[1, 0], axes[1, 2], axes[2, 1]]
        max_ax = axes[1, 1]
        all_ground_truth = []
        indexes = {}
        n_prints = 100
        for i, (r, c, w) in enumerate(all_coords):
            actions_ground_truth = []
            for a in range(4):
                self.r = r
                self.c = c
                self.w = w
                self.step(a)
                ground_truth = self.ground_truth_distances(w)
                # take the movement of the window into account
                dw = self.w - w
                actions_ground_truth.append(ground_truth)
            all_ground_truth.append(actions_ground_truth)
            # render
            if (i + 1) % (n // n_prints) == 0:
                print('{}%'.format(round(100 * (i + 1) / n)))
                for a in range(4):
                    ac_axes[a].clear()
                    ac_axes[a].imshow(actions_ground_truth[a], 'inferno')
                max_ax.clear()
                max_ax.imshow(np.stack(actions_ground_truth, axis=2).max(2), 'inferno')
                fig.canvas.draw()
        all_ground_truth = np.array(all_ground_truth)
        all_ground_truth = np.moveaxis(all_ground_truth, 1, -1)
        gt_path = '{}/gridworld_gound_truth_{}'.format(path, self.level)
        np.save(gt_path, all_ground_truth)
        print('Q-frames saved in {}'.format(gt_path))

        plt.close(fig)
