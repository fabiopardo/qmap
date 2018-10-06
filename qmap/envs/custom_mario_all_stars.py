from collections import deque
import gym
import retro
from gym import Env, spaces, utils
from gym.utils import seeding
import numpy as np
from scipy.misc import imresize
from termcolor import colored


actions = {
    #              jump/swim
    #              |  run/shoot
    #              |  |        up/down/pipe/crouch
    #              |  |        |     left/rigth
    #              +--|--------|-----|-----+
    #              |  +--------|-----|-----|--+
    #              |  |        +--+  |     |  |
    #              |  |        |  |  +--+  |  |
    #              |  |        |  |  |  |  |  |
    #              B  Y  Se St U  D  L  R  A  X  L  R
    'NOOP':       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'JUMP':       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'JUMP+SHOOT': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'JUMP+LEFT':  [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'JUMP+RIGHT': [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'SHOOT':      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'RUN+LEFT':   [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'RUN+RIGHT':  [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'UP':         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'DOWN':       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'LEFT':       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'RIGHT':      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
}


class Spec:
    def __init__(self, id):
        self.id = id


class LazyFrames(object):
    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    @property
    def shape(self):
        shape = self._frames[0].shape
        return (shape[0], shape[1], shape[2] * len(self._frames))


class CustomSuperMarioAllStarsEnv(Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, screen_ratio=4, coords_ratio=4, use_color=True, use_rc_frame=True, stack=3, frame_skip=4, action_repeat=4, state='marioStars_level_1_1.state'):
        self.env = retro.make('SuperMarioAllStars-Snes', state=state)
        self.screen_ratio = screen_ratio
        self.coords_ratio = coords_ratio
        assert coords_ratio % screen_ratio == 0, (coords_ratio, screen_ratio)
        self.coords_screen_ratio = coords_ratio // screen_ratio
        self.use_color = use_color
        self.use_rc_frame = use_rc_frame
        self.stack = stack
        self.frame_skip = frame_skip
        self.action_repeat = action_repeat
        self.action_names = ['JUMP+LEFT', 'JUMP', 'JUMP+RIGHT', 'LEFT', 'NOOP', 'RIGHT']
        self.action_list = [actions[n] for n in self.action_names]
        n_actions = len(self.action_list)
        self.action_space = spaces.Discrete(n_actions)
        self.original_height = 224
        self.original_width = 256
        self.screen_height = self.original_height // screen_ratio
        self.screen_width = self.original_width // screen_ratio
        self.screen_size = self.screen_height * self.screen_width
        self.screen_shape = (self.screen_height, self.screen_width)
        self.coords_height = self.original_height // coords_ratio
        self.coords_width = self.original_width // coords_ratio
        self.coords_shape = (self.coords_height, self.coords_width)
        n_frames = stack * (3 * use_color + 1 * (not use_color) + use_rc_frame)
        self.frames = deque([], maxlen=(self.frame_skip * (self.stack - 1) + 1))
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, n_frames))
        self.viewer = None
        frame_name = 'RGB' if use_color else 'G'
        if use_rc_frame: frame_name += 'C'
        self.name = 'CustomSuperMarioAllStars_obs{}x{}x{}x{}_skip{}_repeat{}-v0'.format(
            *self.screen_shape, frame_name, stack, frame_skip, action_repeat)
        self.spec = Spec(self.name)

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)
        self.env.seed(seed)

    def reset(self):
        self.score = 0
        screen = self.env.reset()
        game_frame, rcw, full_rc, gameover, flag, _ = self._get_obs()
        assert not gameover and not flag
        for _ in range(self.frames.maxlen):
            self.frames.append(game_frame)
        game_frames = LazyFrames(list(self.frames)[::self.frame_skip])
        return (game_frames, rcw, screen, full_rc)

    def step(self, a):
        ac = self.action_list[a]
        rew = 0
        for _ in range(self.action_repeat):
            screen, rew_, done, info = self.env.step(ac)
            assert rew_ == 0
            game_frame, rcw, full_rc, gameover, flag, rew_ = self._get_obs()
            rew += rew_
            if gameover and not done: # died from timeout
                # rew -= 5 # TODO
                done = True
            if done: break
        self.frames.append(game_frame)
        game_frames = LazyFrames(list(self.frames)[::self.frame_skip])
        return (game_frames, rcw, screen, full_rc), rew, done, info

    def _get_image(self):
        img = self.env.get_screen()
        if not self.use_color:
            img = np.dot(img, [0.299, 0.587, 0.114])
        img = imresize(img, (self.screen_height, self.screen_width), interp='bilinear')
        if not self.use_color:
            img = img[:,:,None]
        return img

    def _get_obs(self):
        # observation
        game_frame = self._get_image()
        full_r = self.env.data.memory.extract(3999, "|u1")
        c0 = self.env.data.memory.extract(537, "|u1")
        c1 = self.env.data.memory.extract(120, "|u1")
        w0 = self.env.data.memory.extract(66, "|u1")
        w1 = self.env.data.memory.extract(67, "|u1")
        gameover = self.env.data.memory.extract(1827, "|u1") != 0
        self.full_c = c0 + 256*c1
        flag = self.full_c == 3161 # TEMP: might work only for level 1
        assert not flag or gameover
        w = w0 + 256*w1
        r, c, w = full_r // self.coords_ratio, self.full_c // self.coords_ratio, w // self.coords_ratio
        if r >= self.coords_height: # TODO: check
            print(colored('/////////////////////////////////', 'red'))
            print(colored('MARIO IS OUT OF BOUNDS! GAMEOVER! {} WHILE MAX IS {}'.format(r, self.coords_height-1), 'red'))
            print(colored('/////////////////////////////////', 'red'))
            r = self.coords_height - 1
            gameover = True
        # rc frame
        if self.use_rc_frame:
            rc_frame = np.zeros(self.screen_shape + (1,), dtype=np.uint8)
            rc_frame[r*self.coords_screen_ratio:(r+1)*self.coords_screen_ratio,
                     (c-w)*self.coords_screen_ratio:(c-w+1)*self.coords_screen_ratio] = 255
            game_frame = np.concatenate((rc_frame, game_frame), axis=2)
        # reward
        score10    = 10    * self.env.data.memory.extract(2003, "|u1")
        score100   = 100   * self.env.data.memory.extract(2002, "|u1")
        score1000  = 1000  * self.env.data.memory.extract(2001, "|u1")
        score10000 = 10000 * self.env.data.memory.extract(2000, "|u1")
        score = score10 + score100 + score1000 + score10000
        rew = score - self.score
        self.score = score
        if flag:
            print(colored('////////////////', 'green'))
            print(colored('FLAG REACHED! :D', 'green'))
            print(colored('////////////////', 'green'))
            rew += 5000 # final bonus (seems to be 400 and then a series of 200 then 150 originaly)
        rew /= 100 # rescale
        return game_frame, (r, c, w), (full_r, self.full_c), gameover, flag, rew

    def render(self, mode='human', close=False):
        return self.env.render(mode=mode)
