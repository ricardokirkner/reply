
class ReplyRunner():
    def __init__(self, agent, env, experiment):
        self.agent = agent
        self.env = env
        self.experiment = experiment
        self.experiment.set_glue_experiment(self)

        self.last_action = None

    def init(self):
        return self.agent.init(self.env.init())

    def start(self):
        observation = self.env.start()
        action = self.agent.start(observation)
        self.last_action = action
        return observation, action

    def step(self):
        rot = self.env.step(self.last_action)
        reward = rot["reward"]
        terminal = rot["terminal"]
        observation = rot.copy()
        del observation["reward"]
        del observation["terminal"]

        if terminal:
           self.agent.end(reward)
           return rot
        else:
            action = self.agent.step(reward, observation)
            self.last_action = action
        roat = rot.copy()
        roat["action"] = action
        return reward, observation, terminal, action

    def cleanup(self):
        self.env.cleanup()
        self.agent.cleanup()

    def run(self):
        self.experiment.run()
