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
        self.initialized = True

    def start(self, observation):
        return {}

    def step(self, reward, observation):
        return {}

    def end(self, reward):
        pass

    def cleanup(self):
        pass

class LearningAgent(Agent):

    def start(self, observation):
        self.learner.new_episode()
        action = self.learner.policy.select_action(observation)
        self.last_observation = observation
        self.last_action = action
        return action

    def step(self, reward, observation):
        self.learner.update(self.last_observation, self.last_action, reward,
                            observation)
        action = self.learner.policy.select_action(observation)
        self.last_observation = observation
        self.last_action = action
        return action

    def end(self, reward):
        self.learner.update(self.last_observation, self.last_action, reward,
                            None)
