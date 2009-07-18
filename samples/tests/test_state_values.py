import unittest
import random

from reply.runner import Run
from reply.experiment import Experiment
from reply.learner import SarsaLearner

from .samples.state_value import StateValueAgent, StateValueEnvironment

class TestStateValue(unittest.TestCase):
    agent_class = StateValueAgent
    def test_run(self):
        # use fixed sequence for random
        random.seed(1)

        agent = self.agent_class()
        env = StateValueEnvironment()
        outterself = self

        class TestExperiment(Experiment):
            def run(self):
                self.init()
                for i in xrange(10000):
                    self.episode()
                self.cleanup()
                data = agent.learner.policy.storage.data.transpose()
                error = sum(sum(abs(
                    env.ps - data)))
                outterself.assert_(error < 0.14, "Error too big: %s"%(error))

        r = Run()
        r.run(agent, env, TestExperiment())

        #return random to random
        random.seed()

class SarsaStateValueAgent(StateValueAgent):
    learner_class = SarsaLearner

class SarsaStateValue(TestStateValue):
    agent_class = SarsaStateValueAgent


if __name__ == '__main__':
    unittest.main()