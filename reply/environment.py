import simplejson

from reply.types import Space
from reply.util import MessageHandler, TaskSpec

class Environment(MessageHandler):
    problem_type = "episodic"
    discount_factor = 1.0
    rewards = (0, 1)
    actions_spec = {}
    observations_spec = {}

    def __init__(self):
        super(Environment, self).__init__()
        self.initialized = False
        self.set_action_space(**self.actions_spec)
        self.set_observation_space(**self.observations_spec)

    def set_observation_space(self, **kwargs):
        if self.initialized:
            raise Exception("Can't change observation space after init")
        self._observation_space = Space(kwargs)

    def set_action_space(self, **kwargs):
        if self.initialized:
            raise Exception("Can't change action space after init")
        self._action_space = Space(kwargs)

    def get_task_spec(self):
        task_spec = TaskSpec(problem_type='episodic',
                             discount_factor=self.discount_factor,
                             observations=self._observation_space,
                             actions=self._action_space,
                             rewards=self.rewards,
                             extra=self.__doc__)
        return task_spec

    #
    # Standard API
    #

    def init(self):
        self._init()
        self.initialized = True
        return self.get_task_spec()

    def start(self):
        observation = self._start()
        self.started = True
        return observation

    def step(self, action):
        observation = self._step(action)
        return observation

    def cleanup(self):
        self._cleanup()

    #
    # Overridable Methods
    #

    def _init(self):
        pass

    def _start(self):
        pass

    def _step(self, action):
        pass

    def _cleanup(self):
        pass

