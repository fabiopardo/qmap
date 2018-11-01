from collections import deque
import gym
from gym import Env, spaces, utils
from gym.utils import seeding
import numpy as np
from scipy.misc import imresize

from qmap.utils.lazy_frames import LazyFrames


actions = {
    'NOOP':           0,
    'FIRE':           1, # useless?
    'UP':             2,
    'RIGHT':          3,
    'LEFT':           4,
    'DOWN':           5,
    'UPRIGHT':        6, # useless
    'UPLEFT':         7, # useless
    'DOWNRIGHT':      8, # useless
    'DOWNLEFT':       9, # useless
    'UPFIRE':        10, # useless?
    'RIGHTFIRE':     11,
    'LEFTFIRE':      12,
    'DOWNFIRE':      13, # useless
    'UPRIGHTFIRE':   14, # useless
    'UPLEFTFIRE':    15, # useless
    'DOWNRIGHTFIRE': 16, # useless
    'DOWNLEFTFIRE':  17, # useless
}


class CustomMontezumaEnv(Env, utils.EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, screen_ratio=4, coords_ratio=4, use_color=True, use_rc_frame=True, stack=3, frame_skip=4, action_repeat=4):
        utils.EzPickle.__init__(self, 'montezuma_revenge', 'image')
        self.env = gym.make('MontezumaRevengeNoFrameskip-v4').unwrapped
        self.ale = self.env.ale
        self.ale.setFloat('repeat_action_probability'.encode('utf-8'), 0) # deterministic
        self.max_lives = self.ale.lives()
        # observations
        self.screen_ratio = screen_ratio
        self.original_height = 224
        self.original_width = 160
        self.screen_height = self.original_height // screen_ratio
        self.screen_width = self.original_width // screen_ratio
        self.screen_shape = (self.screen_height, self.screen_width)
        self.use_color = use_color
        self.use_rc_frame = use_rc_frame
        self.stack = stack
        self.frame_skip = frame_skip
        n_frames = stack * (3 * use_color + 1 * (not use_color) + use_rc_frame)
        self.frames = deque([], maxlen=(self.frame_skip * (self.stack - 1) + 1))
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, n_frames))
        # coordinates
        self.coords_ratio = coords_ratio
        assert coords_ratio % screen_ratio == 0, (coords_ratio, screen_ratio)
        self.coords_screen_ratio = coords_ratio // screen_ratio
        self.coords_height = self.original_height // coords_ratio
        self.coords_width = self.original_width // coords_ratio
        self.coords_shape = (self.coords_height, self.coords_width)
        # actions
        self.action_repeat = action_repeat
        self.action_names = ['LEFTFIRE', 'UP', 'RIGHTFIRE', 'LEFT', 'NOOP', 'RIGHT', 'DOWN']
        self.action_list = [actions[n] for n in self.action_names]
        n_actions = len(self.action_list)
        self.action_space = spaces.Discrete(n_actions)
        # miscellaneous
        frame_name = 'RGB' if use_color else 'G'
        if use_rc_frame: frame_name += 'C'
        self.name = 'CustomMontezuma_obs{}x{}x{}x{}_qframes{}x{}x{}_skip{}_repeat{}-v0'.format(
            *self.screen_shape, frame_name, stack, *self.coords_shape, n_actions, frame_skip, action_repeat)

    def seed(self, seed=None):
        return self.env.seed(seed)

    def reset(self):
        self.score = 0
        screen_ = self.env.reset()
        game_frame, rcw, full_rc, gameover = self._get_obs()
        for _ in range(self.frames.maxlen):
            self.frames.append(game_frame)
        game_frames = LazyFrames(list(self.frames)[::self.frame_skip])
        screen = np.zeros((224, 160, 3)) # padding
        screen[7:-7] = screen_
        return (game_frames, rcw, screen, full_rc)

    def step(self, a):
        ac = self.action_list[a]
        rew = 0
        for _ in range(self.action_repeat):
            screen_, rew_, done, info = self.env.step(ac)
            if ac == 1:
                ac = 0 # after a jump only noops
            game_frame, rcw, full_rc, gameover = self._get_obs()
            rew += rew_
            if gameover and not done: # died from timeout
                done = True
            if done: break
        if rew != 0: print('OMG a reward!', rew)
        self.frames.append(game_frame)
        game_frames = LazyFrames(list(self.frames)[::self.frame_skip])
        screen = np.zeros((224, 160, 3)) # padding
        screen[7:-7] = screen_
        return (game_frames, rcw, screen, full_rc), rew, done, info

    def _get_image(self):
        if self.use_color:
            img = np.zeros((224, 160, 3)) # padding
            img[7:-7] = self.ale.getScreenRGB()
            img = imresize(img, (self.screen_height, self.screen_width), interp='bilinear')
        else:
            img = np.zeros((224, 160, 1)) # padding
            img[7:-7] = self.ale.getScreenGrayscale()
            img = imresize(img[:,:,0], (self.screen_height, self.screen_width), interp='bilinear')[:,:,None]
        return img

    def _get_ram(self):
        return to_ram(self.ale)

    def _get_obs(self):
        game_frame = self._get_image()
        ram = self.ale.getRAM()
        full_r, full_c = 320 - ram[43], ram[42]
        r, c, w = full_r // self.coords_ratio, full_c // self.coords_ratio, 0
        # rc frame
        if self.use_rc_frame:
            rc_frame = np.zeros(self.screen_shape + (1,), dtype=np.uint8)
            rc_frame[r*self.coords_screen_ratio:(r+1)*self.coords_screen_ratio,
                     (c-w)*self.coords_screen_ratio:(c-w+1)*self.coords_screen_ratio] = 255
            game_frame = np.concatenate((rc_frame, game_frame), axis=2)
        gameover = self.ale.game_over() or self.ale.lives() != self.max_lives
        return game_frame, (r, c, w), (full_r, full_c), gameover

    def render(self, mode='human', close=False):
        return self.env.render(mode=mode, close=close)
