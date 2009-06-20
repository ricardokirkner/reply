from reply.datatypes import Integer, Model
from reply.util import MessageHandler, TaskSpec

class Environment(MessageHandler):
    problem_type = 'episodic'
    discount_factor = 1.0
    rewards = Integer(0, 1)
    model = Model()

    def __init__(self):
        super(Environment, self).__init__()
        self.initialized = False
        self.started = False

        self.extra = self._get_names()

    def get_task_spec(self):
        task_spec = TaskSpec(problem_type=self.problem_type,
                             discount_factor=self.discount_factor,
                             observations=self.model.observations,
                             actions=self.model.actions,
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
        rot = self._step(action)
        return rot

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
        observations = self.model.observations
        actions = self.model.actions
        extra = ""
        observation_names = observations.get_names_spec()
        action_names = actions.get_names_spec()
        if observation_names:
            extra += "OBSERVATIONS %s" % observation_names
            if action_names:
                extra += " "
        if action_names:
            extra += "ACTIONS %s" % action_names
        return extra
