from reply.datatypes import Space
from reply.util import MessageHandler

class Agent(MessageHandler):

    def __init__(self):
        super(Agent, self).__init__()
        self._observation_space = None
        self._action_space = None
        self.initialized = False

    def set_action_space(self, space=None, **kwargs):
        if self.initialized:
            raise Exception("Can't change action space after init")
        if space is not None:
            self._action_space = space
        else:
            self._action_space = Space(kwargs)

    def set_observation_space(self, space=None, **kwargs):
        if self.initialized:
            raise Exception("Can't change observation space after init")
        if space is not None:
            self._observation_space = space
        else:
            self._observation_space = Space(kwargs)

    #
    # Standard API
    #

    def init(self, task_spec):
        self.set_action_space(task_spec.actions)
        self.set_observation_space(task_spec.observations)
        self._init(task_spec)
        self.initialized = True

    def start(self, observation):
        action = self._start(observation)
        return action

    def step(self, reward, observation):
        action = self._step(reward, observation)
        return action

    def end(self, reward):
        self._end(reward)

    def cleanup(self):
        self._cleanup()

    #
    # Overridable Methods
    #

    def _init(self, task_spec):
        pass

    def _start(self, observation):
        return {}

    def _step(self, reward, observation):
        return {}

    def _end(self, reward):
        pass

    def _cleanup(self):
        pass
