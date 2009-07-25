import unittest
import random

from reply.runner import Run
from reply.experiment import Experiment
from reply.learner import SarsaLearner

from .samples.rock_paper_scissor import RockPaperScissorAgent, \
                                        RockPaperScissorEnvironment, \
                                        PAPER


class TestRockPaperScissor(unittest.TestCase):
    agent_class = RockPaperScissorAgent
    expected_mappings = [({'state': 0}, {'play': PAPER})]
    def test_run(self):
        # use fixed sequence for random
        random.seed(1)

        agent = self.agent_class()
        env = RockPaperScissorEnvironment()
        outerself = self

        class TestExperiment(Experiment):
            def run(self):
                self.init()
                for i in xrange(10000):
                    self.episode()
                self.cleanup()
                mappings = agent.policy.get_mappings()
                import pprint
                pprint.pprint(mappings)
                pprint.pprint(outerself.expected_mappings)
                outerself.assertEqual(mappings, outerself.expected_mappings)
                print agent.storage.data

        r = Run()
        r.run(agent, env, TestExperiment())

        #return random to random
        random.seed()


class SarsaRockPaperScissorAgent(RockPaperScissorAgent):
    learner_class = SarsaLearner


class TestSarsaRockPaperScissor(TestRockPaperScissor):
    agent_class = SarsaRockPaperScissorAgent
    expected_mappings = [({'state': 0}, {'play': PAPER})]


if __name__ == '__main__':
    unittest.main()
