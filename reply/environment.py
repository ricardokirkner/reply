from reply.datatypes import Double, Model
from reply.util import MessageHandler, TaskSpec

class Environment(MessageHandler):

    """An Environment is an abstraction of the problem's environment.

    Attributes:

    - problem_type -- describes the problem type: episodic | continuous

    - discount_factor -- the discount value used for this environment

    - rewards -- a Double representing the min and max rewards

    - model -- a Model instance

    """

    problem_type = 'episodic'
    discount_factor = 1.0
    rewards = Double(0, 1)
    model = Model()

    def __init__(self):
        """Create an Environment instance."""
        super(Environment, self).__init__()
        self.initialized = False

        self.extra = self._get_names()

    def get_task_spec(self):
        """Return a TaskSpec object describing the problem."""
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
        """Initialize the environment.

        Return a TaskSpec describing it.

        """
        self.initialized = True
        return self.get_task_spec()

    def start(self):
        """Start an episode.

        Return an observation.

        """
        observation = {}
        return observation

    def step(self, action):
        """Perform a step in the world.

        Return a reward-observation-terminal dictionary.

        """
        rot = {'reward': 0, 'terminal': False}
        return rot

    def cleanup(self):
        """Cleanup after the episode has ended."""
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
