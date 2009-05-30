from reply.types import Space
from reply.util import MessageHandler, TaskSpec

class Agent(MessageHandler):

    def __init__(self):
        super(Agent, self).__init__()
        self.initialized = False

    def set_action_space(self, **kwargs):
        if self.initialized:
            raise Exception("Can't change action space after init")
        self._action_space = Space(kwargs)

    #
    # Standard API
    #

    def init(self, task_spec):
        task_spec = TaskSpec.parse(task_spec)
        actions_spec = self._get_actions_spec(task_spec.actions.values())
        self.set_action_space(**actions_spec)
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
        pass

    def _step(self, reward, observation):
        pass

    def _end(self, reward):
        pass

    def _cleanup(self):
        pass

    #
    # Helper Methods
    #

    def _get_actions_spec(self, actions):
        def update(x, y):
            x.update(y)
            return x
        actions_spec = reduce(update, actions)
        return actions_spec

