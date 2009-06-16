
class Experiment(object):
    steps = 100
    episodes = 10000

    def __init__(self):
        self.glue_experiment = None

    def set_glue_experiment(self, experiment):
        self.glue_experiment = experiment

    # RL Glue Experiment API

    def init(self):
        return self.glue_experiment.init()

    def start(self):
        return self.glue_experiment.start()

    def step(self):
        return self.glue_experiment.step()

    def episode(self):
        return self.glue_experiment.episode()

    def return_reward(self):
        return self.glue_experiment.return_reward()

    def num_steps(self):
        return self.glue_experiment.num_steps()

    def cleanup(self):
        return self.glue_experiment.cleanup()

    def agent_message(self, message):
        return self.glue_experiment.agent_message(message)

    def env_message(self, message):
        return self.glue_experiment.env_message(message)

    # Method to override

    def run(self):
        self.init()
        for episode in range(self.episodes):
            self.start()
            steps = 0
            terminal = False
            while steps < self.steps and not terminal:
                roat = self.step()
                terminal = roat["terminal"]
                steps += 1
        self.cleanup()
