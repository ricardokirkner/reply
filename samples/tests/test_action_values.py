import unittest
import random

from reply.runner import Run
from reply.experiment import Experiment
from .samples.action_value import ActionValueAgent, ActionValueEnvironment

class TestActionValue(unittest.TestCase):
    def test_run(self):
        # use fixed sequence for random
        random.seed(1)

        agent = ActionValueAgent()
        env = ActionValueEnvironment()
        outterself = self

        class TestExperiment(Experiment):
            def run(self):
                self.init()
                for i in xrange(10000):
                    self.episode()
                self.cleanup()
                error = sum(sum(abs(
                    env.ps - agent.learner.policy.storage.data)))
                print "error", error
                outterself.assert_(error < 0.13, "Error too big: %s"%(error))

        r = Run()
        r.run(agent, env, TestExperiment())

        #return random to random
        random.seed()


if __name__ == '__main__':
    unittest.main()