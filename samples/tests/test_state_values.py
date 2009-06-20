import unittest
import random

from reply.runner import Run
from reply.experiment import Experiment
from .samples.state_value import StateValueAgent, StateValueEnvironment

class TestActionValue(unittest.TestCase):
    def test_run(self):
        # use fixed sequence for random
        random.seed(1)

        agent = StateValueAgent()
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


if __name__ == '__main__':
    unittest.main()