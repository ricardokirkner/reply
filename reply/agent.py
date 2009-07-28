from reply.base import Parameter, PersistingObject
from reply.datatypes import Model
from reply.util import MessageHandler

class Agent(MessageHandler, PersistingObject):
    model = Parameter("The (observations, actions) model")
    policy_class = Parameter("The policy class to be used")
    storage_class = Parameter("The storage class to be used")

    def __init__(self, model=None, policy_class=None, storage_class=None):
        super(Agent, self).__init__()
        if model is not None:
            self.model = model
        if policy_class is not None:
            self.policy_class = policy_class
        if storage_class is not None:
            self.storage_class = storage_class
        self.initialized = False

    def build_model(self, task_spec):
        if not self.initialized:
            if not isinstance(self.model, Model):
                observations = task_spec.observations
                actions = task_spec.actions
                self.model = Model(observations, actions)

    def build_policy(self):
        if not self.initialized:
            self.policy = self.policy_class(self)

    def build_storage(self):
        if not self.initialized:
            self.storage = self.storage_class(self)

    #
    # Standard API
    #

    def init(self, task_spec):
        self.build_model(task_spec)
        self.build_policy()
        self.build_storage()
        self.initialized = True

    def start(self, observation):
        self.policy.on_episode_start()
        action = self.policy.select_action(observation)
        return action

    def step(self, reward, observation):
        action = self.policy.select_action(observation)
        return action

    def end(self, reward):
        pass

    def cleanup(self):
        pass


class LearningAgent(Agent):
    def build_learner(self):
        self.learner = self.learner_class(self)

    def init(self, task_spec):
        super(LearningAgent, self).init(task_spec)
        self.build_learner()

        self.last_observation = None
        self.last_action = None

    def start(self, observation):
        self.learner.on_episode_start()
        action = super(LearningAgent, self).start(observation)
        self.last_observation = observation
        self.last_action = action
        return action

    def step(self, reward, observation):
        self.learner.update(self.last_observation, self.last_action, reward,
                            observation)
        action = super(LearningAgent, self).step(reward, observation)
        self.last_observation = observation
        self.last_action = action
        return action

    def end(self, reward):
        self.learner.update(self.last_observation, self.last_action, reward,
                            None)
