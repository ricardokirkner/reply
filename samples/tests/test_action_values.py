import unittest
import random

from reply.runner import Run
from reply.experiment import Experiment
from reply.learner import SarsaLearner

from .samples.action_value import ActionValueAgent, ActionValueEnvironment, \
        ActionValueExperiment

class TestActionValue(unittest.TestCase):
    agent_class = ActionValueAgent
    def test_run(self):
        # use fixed sequence for random
        random.seed(1)

        agent = self.agent_class()
        env = ActionValueEnvironment()
        outterself = self

        class TestExperiment(ActionValueExperiment):
            def run(self):
                self.init()
                for i in xrange(10000):
                    self.episode()
                self.cleanup()
                error = sum(sum(abs(
                    env.ps - agent.storage.data)))
                outterself.assert_(error < 0.13, "Error too big: %s"%(error))

        r = Run()
        r.run(agent, env, TestExperiment())

        #return random to random
        random.seed()

class SarsaActionValueAgent(ActionValueAgent):
    learner_class = SarsaLearner

class SarsaActionValue(TestActionValue):
    agent_class = SarsaActionValueAgent

if __name__ == '__main__':
    unittest.main()
