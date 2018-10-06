import tensorflow as tf


class Agent(object):
    def __new__(cls, *args, **kwargs):
        # We use __new__ since we want the env author to be able to
        # override __init__ without remembering to call super.
        agent = super(Agent, cls).__new__(cls)
        return agent

    def step(self, *args, **kwargs): raise NotImplementedError
    def reset(self, *args, **kwargs): raise NotImplementedError
    def test(self, *args, **kwargs): raise NotImplementedError

    @property
    def unwrapped(self):
        return self

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)

class Policy(object):
    def __init__(self, name, recurrent=False, prefix='', **kwargs):
        self.recurrent = recurrent
        with tf.variable_scope(name):
            self.scope = tf.get_variable_scope().name
            self._init(prefix=prefix, **kwargs)

    def _init(self):
        raise NotImplementedError

    def act(self, ob, stochastic=True):
        ac1, vpred1 = self._act(*[[o] for o in ob], stochastic)
        if type(ac1) == list:
            ac1 = [x[0] for x in ac1]
        else:
            ac1 = ac1[0]
        return ac1, vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

    # def update_rms(self, obs):
    #     raise NotImplementedError

class AgentWrapper(Agent):
    agent = None

    def __init__(self, agent):
        self.agent = agent

    @classmethod
    def class_name(cls):
        return cls.__name__

    def _ensure_no_double_wrap(self):
        agent = self.agent
        while True:
            if isinstance(agent, Wrapper):
                if agent.class_name() == self.class_name():
                    raise error.DoubleWrapperError("Attempted to double wrap with Wrapper: {}".format(self.__class__.__name__))
                agent = agent.agent
            else:
                break

    def reset(self, *args, **kwargs):
        return self.agent.reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self.agent.step(*args, **kwargs)

    def test(self, *args, **kwargs):
        return self.agent.test(*args, **kwargs)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.agent)

    def __repr__(self):
        return str(self)

    @property
    def unwrapped(self):
        return self.agent.unwrapped
