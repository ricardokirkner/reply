from reply.base import Parameter, PersistingObject
from reply.datatypes import Model
from reply.util import MessageHandler

class Agent(MessageHandler, PersistingObject):

    """Base Agent class.

    An agent has three required parameters:

    - model: A Model instance representing both observation and
             action spaces

    - policy_class: The class to be used for instantiating the
                    policy

    - storage_class: The class to be used for instantiating the
                     storage

    """

    model = Parameter("The (observations, actions) model")
    policy_class = Parameter("The policy class to be used")
    storage_class = Parameter("The storage class to be used")

    def __init__(self, model=None, policy_class=None, storage_class=None):
        """Create an Agent instance.

        All three parameters can be given to the constructor, allowing it
        to override class level parameters.

        """
        super(Agent, self).__init__()
        if model is not None:
            self.model = model
        if policy_class is not None:
            self.policy_class = policy_class
        if storage_class is not None:
            self.storage_class = storage_class
        self.initialized = False

    def build_model(self, task_spec):
        """Create the model instance from a TaskSpec object."""
        if not self.initialized:
            if not isinstance(self.model, Model):
                observations = task_spec.observations
                actions = task_spec.actions
                self.model = Model(observations, actions)

    def build_policy(self):
        """Create the policy instance from policy_class parameter."""
        if not self.initialized:
            self.policy = self.policy_class(self)

    def build_storage(self):
        """Create the storage instance from the storage_class parameter."""
        if not self.initialized:
            self.storage = self.storage_class(self)

    #
    # Standard API
    #

    def init(self, task_spec):
        """Initialize the agent from a TaskSpec object."""
        self.build_model(task_spec)
        self.build_policy()
        self.build_storage()
        self.initialized = True

    def start(self, observation):
        """Start an episode.

        Return an action.

        """
        self.policy.on_episode_start()
        action = self.policy.select_action(observation)
        return action

    def step(self, reward, observation):
        """Perform a step in the world.

        Return an action.

        """
        action = self.policy.select_action(observation)
        return action

    def end(self, reward):
        """End an episode."""
        pass

    def cleanup(self):
        """Cleanup after the episode has ended."""
        pass


class LearningAgent(Agent):

    """Base class for agents with learning capabilities.

    A learning agent has all required parameters an Agent has, plus:

    - learner_class: The class to be used for instantiating the
                     learner

    """

    learner_class = Parameter("The learner class to be used")

    def build_learner(self):
        """Create the learner instance from the learner_class parameter."""
        self.learner = self.learner_class(self)

    def init(self, task_spec):
        """Initialize the agent from a TaskSpec object."""
        super(LearningAgent, self).init(task_spec)
        self.build_learner()

        self.last_observation = None
        self.last_action = None

    def start(self, observation):
        """Start an episode.

        Return an action.

        """
        self.learner.on_episode_start()
        action = super(LearningAgent, self).start(observation)
        self.last_observation = observation
        self.last_action = action
        return action

    def step(self, reward, observation):
        """Perform a step in the world.

        Return an action.

        """
        self.learner.update(self.last_observation, self.last_action, reward,
                            observation)
        action = super(LearningAgent, self).step(reward, observation)
        self.last_observation = observation
        self.last_action = action
        return action

    def end(self, reward):
        """End an episode."""
        self.learner.update(self.last_observation, self.last_action, reward,
                            None)
