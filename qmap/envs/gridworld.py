import numpy as np
from gym import Env, spaces
from gym.utils import seeding
from gym.envs.registration import register
import math
import matplotlib.pyplot as plt
import os


class Spec:
    def __init__(self, id):
        self.id = id

class GridWorld(Env):
    def __init__(self, level='level1', window_size=32, scale=1):
        if not '.' in level: level += '.bmp'
        self.name = 'GridWorld_window{}_scale{}'.format(window_size, scale)
        self.window_size = window_size
        self.scale = scale
        self.padding = window_size//2-1
        self.walls = np.logical_not(plt.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), level)))
        self.padded_walls = np.logical_not(np.pad(np.logical_not(self.walls), ((0, 0), (self.padding, self.padding)), 'constant'))
        self.available_coords = np.array(np.where(np.logical_not(self.walls))).transpose()
        height = self.walls.shape[0]
        self.metadata = {'render.modes': ['human', 'rgb_array']}
        self.rows, self.cols = self.walls.shape
        self.coords_shape = (height // scale, window_size // scale)
        self.observation_space = spaces.Box(0, 255, (height, window_size, 3), dtype=np.float32)
        self.n_actions = 4
        self.action_space = spaces.Discrete(self.n_actions)
        self.viewer = None
        self.spec = Spec(self.name+'-v0')
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
        frames = np.zeros((self.rows, self.window_size, 3), dtype=np.uint8)
        frames[:, :, 2] = self.padded_walls[:, self.padding+self.w:self.padding+self.w+self.window_size] * 255
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
