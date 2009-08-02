
class Experiment(object):

    """An Experiment describes the experiment to be performed.

    The Experiment is actually just a proxy to the real experiment instance.
    It defines the common API all experiments must implement.

    An experiment involves an Agent performing actions within an
    Environment.

    Attributes:

    - max_steps -- the max number of steps to perform on each episode

    - max_episodes -- the max number of episodes to perform

    """

    max_steps = 100
    max_episodes = 10000

    def __init__(self):
        """Create an Experiment instance."""
        self.glue_experiment = None

    def set_glue_experiment(self, experiment):
        """Bind the Experiment instance to the real experiment."""
        self.glue_experiment = experiment

    # RL Glue Experiment API

    def init(self):
        """Initialize the experiment.

        Return the TaskSpec provided by the real experiment.

        Raise a NotImplementedError if a real experiment has not been binded.

        """
        if self.glue_experiment is None:
            raise NotImplementedError(
                "Glue Experiment has not been initialized yet.")
        return self.glue_experiment.init()

    def start(self):
        """Start an episode.

        Return the observation provided by the real experiment.

        Raise a NotImplementedError if a real experiment has not been binded.

        """
        if self.glue_experiment is None:
            raise NotImplementedError(
                "Glue Experiment has not been initialized yet.")
        return self.glue_experiment.start()

    def step(self):
        """Perform a step in the world.

        Return the observation provided by the real experiment.

        Raise a NotImplementedError if a real experiment has not been binded.

        """
        if self.glue_experiment is None:
            raise NotImplementedError(
                "Glue Experiment has not been initialized yet.")
        return self.glue_experiment.step()

    def episode(self):
        """FIXME: Find out what this method is about.

        Raise a NotImplementedError if a real experiment has not been binded.

        """
        if self.glue_experiment is None:
            raise NotImplementedError(
                "Glue Experiment has not been initialized yet.")
        return self.glue_experiment.episode()

    def return_reward(self):
        """Return the accumulated reward during the experiment.

        Raise a NotImplementedError if a real experiment has not been binded.

        """
        if self.glue_experiment is None:
            raise NotImplementedError(
                "Glue Experiment has not been initialized yet.")
        return self.glue_experiment.return_reward()

    def num_steps(self):
        """Return the number of steps performed during the experiment.

        Raise a NotImplementedError if a real experiment has not been binded.

        """
        if self.glue_experiment is None:
            raise NotImplementedError(
                "Glue Experiment has not been initialized yet.")
        return self.glue_experiment.num_steps()

    def cleanup(self):
        """Cleanup after the experiment has ended.

        Raise a NotImplementedError if a real experiment has not been binded.

        """
        if self.glue_experiment is None:
            raise NotImplementedError(
                "Glue Experiment has not been initialized yet.")
        return self.glue_experiment.cleanup()

    def agent_call(self, function_name, *args, **kwargs):
        """Call a method on the agent, and return its outcome.

        Raise a NotImplementedError if a real experiment has not been binded.

        """
        if self.glue_experiment is None:
            raise NotImplementedError(
                "Glue Experiment has not been initialized yet.")
        return self.glue_experiment.agent_call(function_name, *args, **kwargs)

    def agent_message(self, message):
        """Send a message to the agent, and return its response.

        Raise a NotImplementedError if a real experiment has not been binded.

        """
        if self.glue_experiment is None:
            raise NotImplementedError(
                "Glue Experiment has not been initialized yet.")
        return self.glue_experiment.agent_message(message)

    def env_call(self, function_name, *args, **kwargs):
        """Call a method on the environment, and return its outcome.

        Raise a NotImplementedError if a real experiment has not been binded.

        """
        if self.glue_experiment is None:
            raise NotImplementedError(
                "Glue Experiment has not been initialized yet.")
        return self.glue_experiment.env_call(function_name, *args, **kwargs)

    def env_message(self, message):
        """Send a message to the environment, and return its responde.

        Raise a NotImplementedError if a real experiment has not been binded.

        """
        if self.glue_experiment is None:
            raise NotImplementedError(
                "Glue Experiment has not been initialized yet.")
        return self.glue_experiment.env_message(message)

    # Method to override

    def run(self):
        """Run the experiment."""
        self.init()
        for episode in range(self.max_episodes):
            self.start()
            steps = 0
            terminal = False
            while steps < self.max_steps and not terminal:
                roat = self.step()
                terminal = roat["terminal"]
                steps += 1
        self.cleanup()
