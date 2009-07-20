from reply.datatypes import Model
from reply.util import MessageHandler

class Agent(MessageHandler):
    model = None

    def __init__(self):
        super(Agent, self).__init__()
        self.initialized = False

    #
    # Standard API
    #

    def init(self, task_spec):
        if self.model is None:
            self.model = Model(task_spec.observations, task_spec.actions)
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
    def build_storage(self):
        self.storage = self.storage_class(self)

    def build_policy(self):
        self.policy = self.policy_class(self)

    def build_learner(self):
        self.learner = self.learner_class(self)

    def init(self, task_spec):
        self.build_storage()
        self.build_policy()
        self.build_learner()

        self.last_observation = None
        self.last_action = None

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
