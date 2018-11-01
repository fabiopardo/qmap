from gym import Wrapper

from qmap.utils.csv_logger import CSVLogger


class PerfLogger(Wrapper):
    def __init__(self, env, gamma, path):
        super(PerfLogger, self).__init__(env)
        self.gamma = gamma
        self.episodes = 0
        self.steps = 0
        self.discount = 1
        self.ep_length = 0
        self.needs_reset = True
        self.logger = CSVLogger(['steps', 'undiscounted return', 'discounted return', 'episode length'], path + '/score')

    def step(self, action):
        assert not self.needs_reset, 'The environment should be reset.'
        self.steps += 1
        self.ep_length += 1
        observation, reward, done, info = self.env.step(action)
        self.undiscounted_return += reward
        self.discounted_return += self.discount * reward
        self.discount *= self.gamma
        if done: self.needs_reset = True
        return observation, reward, done, info

    def reset(self):
        if self.ep_length != 0:
            self.logger.log(self.steps-self.ep_length, self.undiscounted_return, self.discounted_return, self.ep_length)
        self.needs_reset = False
        self.undiscounted_return = 0
        self.discounted_return = 0
        self.ep_length = 0
        self.episodes += 1
        self.discount = 1
        return self.env.reset()
