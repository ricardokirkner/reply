import simplejson

from reply.types import Integer, Space
from reply.util import MessageHandler, TaskSpec

class Environment(MessageHandler):
    problem_type = 'episodic'
    discount_factor = 1.0
    rewards = Integer(0, 1)
    actions_spec = {}
    observations_spec = {}

    def __init__(self):
        super(Environment, self).__init__()
        self.initialized = False
        self.set_action_space(**self.actions_spec)
        self.set_observation_space(**self.observations_spec)
        self.extra = self._get_names()

    def set_observation_space(self, space=None, **kwargs):
        if self.initialized:
            raise Exception("Can't change observation space after init")
        if space is not None:
            self._observation_space = space
        else:
            self._observation_space = Space(kwargs)

    def set_action_space(self, space=None, **kwargs):
        if self.initialized:
            raise Exception("Can't change action space after init")
        if space is not None:
            self._action_space = space
        else:
            self._action_space = Space(kwargs)

    def get_task_spec(self):
        task_spec = TaskSpec(problem_type=self.problem_type,
                             discount_factor=self.discount_factor,
                             observations=self._observation_space,
                             actions=self._action_space,
                             rewards=self.rewards,
                             extra=self.extra)
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
        return {}

    def _step(self, action):
        return {}

    def _cleanup(self):
        pass

    #
    # Helper Methods
    #

    def _get_names(self):
        observations = self._observation_space
        actions = self._action_space
        extra = ""
        observation_names = observations.get_names()
        action_names = actions.get_names()
        if observation_names:
            extra += "OBSERVATIONS %s" % observation_names
            if action_names:
                extra += " "
        if action_names:
            extra += "ACTIONS %s" % action_names
        return extra
