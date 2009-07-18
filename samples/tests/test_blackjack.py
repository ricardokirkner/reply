import unittest
import random

from reply.runner import Run
from reply.experiment import Experiment
from reply.learner import SarsaLearner

from .samples.blackjack import BlackJackAgent, BlackJackEnvironment


HIT, STAND = range(2)


class TestBlackJack(unittest.TestCase):
    agent_class = BlackJackAgent
    expected_mappings = [({'total_points': 2}, {'play': HIT}),
                         ({'total_points': 3}, {'play': HIT}),
                         ({'total_points': 4}, {'play': HIT}),
                         ({'total_points': 5}, {'play': HIT}),
                         ({'total_points': 6}, {'play': HIT}),
                         ({'total_points': 7}, {'play': HIT}),
                         ({'total_points': 8}, {'play': HIT}),
                         ({'total_points': 9}, {'play': HIT}),
                         ({'total_points': 10}, {'play': HIT}),
                         ({'total_points': 11}, {'play': HIT}),
                         ({'total_points': 12}, {'play': HIT}),
                         ({'total_points': 13}, {'play': HIT}),
                         ({'total_points': 14}, {'play': HIT}),
                         ({'total_points': 15}, {'play': STAND}),
                         ({'total_points': 16}, {'play': HIT}),
                         ({'total_points': 17}, {'play': STAND}),
                         ({'total_points': 18}, {'play': STAND}),
                         ({'total_points': 19}, {'play': STAND}),
                         ({'total_points': 20}, {'play': STAND}),
                         ({'total_points': 21}, {'play': STAND}),
                         ({'total_points': 22}, {'play': HIT})]
    def test_run(self):
        # use fixed sequence for random
        random.seed(1)

        agent = self.agent_class()
        env = BlackJackEnvironment()
        outerself = self

        class TestExperiment(Experiment):
            def run(self):
                self.init()
                for i in xrange(10000):
                    self.episode()
                self.cleanup()
                mappings = agent.learner.policy.get_mappings()
                import pprint
                pprint.pprint(mappings)
                pprint.pprint(outerself.expected_mappings)
                outerself.assertEqual(mappings, outerself.expected_mappings)
                outerself.assertEqual(
                    agent.learner.policy.storage.get({'total_points': 22},
                                                      {'play': HIT}), 0)
                outerself.assertEqual(
                    agent.learner.policy.storage.get({'total_points': 22},
                                                      {'play': STAND}), 0)

        r = Run()
        r.run(agent, env, TestExperiment())

        #return random to random
        random.seed()


class SarsaBlackJackAgent(BlackJackAgent):
    learner_class = SarsaLearner


class TestSarsaBlackJack(TestBlackJack):
    agent_class = SarsaBlackJackAgent
    expected_mappings = [({'total_points': 2}, {'play': HIT}),
                         ({'total_points': 3}, {'play': HIT}),
                         ({'total_points': 4}, {'play': HIT}),
                         ({'total_points': 5}, {'play': HIT}),
                         ({'total_points': 6}, {'play': HIT}),
                         ({'total_points': 7}, {'play': HIT}),
                         ({'total_points': 8}, {'play': HIT}),
                         ({'total_points': 9}, {'play': HIT}),
                         ({'total_points': 10}, {'play': HIT}),
                         ({'total_points': 11}, {'play': HIT}),
                         ({'total_points': 12}, {'play': HIT}),
                         ({'total_points': 13}, {'play': HIT}),
                         ({'total_points': 14}, {'play': STAND}),
                         ({'total_points': 15}, {'play': HIT}),
                         ({'total_points': 16}, {'play': STAND}),
                         ({'total_points': 17}, {'play': STAND}),
                         ({'total_points': 18}, {'play': STAND}),
                         ({'total_points': 19}, {'play': STAND}),
                         ({'total_points': 20}, {'play': STAND}),
                         ({'total_points': 21}, {'play': STAND}),
                         ({'total_points': 22}, {'play': HIT})]


if __name__ == '__main__':
    unittest.main()
