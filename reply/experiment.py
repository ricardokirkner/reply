import simplejson

class Experiment(object):
    max_steps = 100
    max_episodes = 10000

    def __init__(self):
        self.glue_experiment = None

    def set_glue_experiment(self, experiment):
        self.glue_experiment = experiment

    # RL Glue Experiment API

    def init(self):
        if self.glue_experiment is None:
            raise NotImplementedError(
                "Glue Experiment has not been initialized yet.")
        return self.glue_experiment.init()

    def start(self):
        if self.glue_experiment is None:
            raise NotImplementedError(
                "Glue Experiment has not been initialized yet.")
        return self.glue_experiment.start()

    def step(self):
        if self.glue_experiment is None:
            raise NotImplementedError(
                "Glue Experiment has not been initialized yet.")
        return self.glue_experiment.step()

    def episode(self):
        if self.glue_experiment is None:
            raise NotImplementedError(
                "Glue Experiment has not been initialized yet.")
        return self.glue_experiment.episode()

    def return_reward(self):
        if self.glue_experiment is None:
            raise NotImplementedError(
                "Glue Experiment has not been initialized yet.")
        return self.glue_experiment.return_reward()

    def num_steps(self):
        if self.glue_experiment is None:
            raise NotImplementedError(
                "Glue Experiment has not been initialized yet.")
        return self.glue_experiment.num_steps()

    def cleanup(self):
        if self.glue_experiment is None:
            raise NotImplementedError(
                "Glue Experiment has not been initialized yet.")
        return self.glue_experiment.cleanup()

    def agent_call(self, function_name, *args, **kwargs):
        if self.glue_experiment is None:
            raise NotImplementedError(
                "Glue Experiment has not been initialized yet.")
        return self.glue_experiment.agent_call(function_name,
                                               *args, **kwargs)

    def agent_message(self, message):
        if self.glue_experiment is None:
            raise NotImplementedError(
                "Glue Experiment has not been initialized yet.")
        return self.glue_experiment.agent_message(message)

    def env_call(self, function_name, *args, **kwargs):
        if self.glue_experiment is None:
            raise NotImplementedError(
                "Glue Experiment has not been initialized yet.")
        return self.glue_experiment.env_call(function_name,
                                             *args, **kwargs)

    def env_message(self, message):
        if self.glue_experiment is None:
            raise NotImplementedError(
                "Glue Experiment has not been initialized yet.")
        return self.glue_experiment.env_message(message)

    # Method to override

    def run(self):
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
