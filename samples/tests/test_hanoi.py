import unittest
import random

from reply.runner import Run
from reply.experiment import Experiment
from reply.learner import SarsaLearner

from .samples.hanoi import HanoiAgent, HanoiEnvironment


class TestHanoi(unittest.TestCase):
    agent_class = HanoiAgent
    expected_mappings = [({'disc_0': 0, 'disc_1': 0, 'disc_2': 0}, {'from_peg': 0, 'to_peg': 2}),
                         ({'disc_0': 2, 'disc_1': 0, 'disc_2': 0}, {'from_peg': 0, 'to_peg': 1}),
                         ({'disc_0': 2, 'disc_1': 1, 'disc_2': 0}, {'from_peg': 2, 'to_peg': 1}),
                         ({'disc_0': 1, 'disc_1': 1, 'disc_2': 0}, {'from_peg': 0, 'to_peg': 2}),
                         ({'disc_0': 1, 'disc_1': 1, 'disc_2': 2}, {'from_peg': 1, 'to_peg': 0}),
                         ({'disc_0': 0, 'disc_1': 1, 'disc_2': 2}, {'from_peg': 1, 'to_peg': 2}),
                         ({'disc_0': 0, 'disc_1': 2, 'disc_2': 2}, {'from_peg': 0, 'to_peg': 2})]


    def test_run(self, num_episodes=150000):
        # use fixed sequence for random
        random.seed(1)

        agent = self.agent_class()
        env = HanoiEnvironment()
        outerself = self

        class TestExperiment(Experiment):
            def run(self):
                self.init()
                for i in xrange(num_episodes):
                    self.episode()
                self.cleanup()
                mappings = agent.learner.policy.get_mappings()
                import pprint
                pprint.pprint(mappings)
                pprint.pprint(outerself.expected_mappings)
                match_expected = True
                for item in outerself.expected_mappings:
                    match_expected = match_expected and item in mappings
                outerself.assertTrue(match_expected)

        r = Run()
        r.run(agent, env, TestExperiment())

        #return random to random
        random.seed()


class SarsaHanoiAgent(HanoiAgent):
    learner_class = SarsaLearner


class TestSarsaHanoi(TestHanoi):
    agent_class = SarsaHanoiAgent
    expected_mappings = [({'disc_0': 0, 'disc_1': 0, 'disc_2': 0}, {'from_peg': 0, 'to_peg': 2}),
                         ({'disc_0': 2, 'disc_1': 0, 'disc_2': 0}, {'from_peg': 0, 'to_peg': 1}),
                         ({'disc_0': 2, 'disc_1': 1, 'disc_2': 0}, {'from_peg': 2, 'to_peg': 1}),
                         ({'disc_0': 1, 'disc_1': 1, 'disc_2': 0}, {'from_peg': 0, 'to_peg': 2}),
                         ({'disc_0': 1, 'disc_1': 1, 'disc_2': 2}, {'from_peg': 1, 'to_peg': 0}),
                         ({'disc_0': 0, 'disc_1': 1, 'disc_2': 2}, {'from_peg': 1, 'to_peg': 2}),
                         ({'disc_0': 0, 'disc_1': 2, 'disc_2': 2}, {'from_peg': 0, 'to_peg': 2})]

    def test_run(self):
        super(TestSarsaHanoi, self).test_run(num_episodes=300000)


if __name__ == '__main__':
    unittest.main()
