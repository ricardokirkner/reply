import unittest

from reply.agent import Agent
from reply.types import Char, Double, Integer, Space


class TestAgent(unittest.TestCase):

    def setUp(self):
        self.agent = Agent()
        self.task_spec = "VERSION RL-Glue-3.0 PROBLEMTYPE episodic DISCOUNTFACTOR 1.0" \
            "OBSERVATIONS INTS (0 1) ACTIONS INTS (0 1) REWARDS (0 1) " \
            "EXTRA OBSERVATIONS INTS o1 ACTIONS INTS a1"

    def test_agent_builder(self):
        self.assertEqual(self.agent.initialized, False)

    def test_agent_set_action_space_already_initialized(self):
        self.agent.init(self.task_spec)
        spec = {'a1': Integer(0, 1)}
        self.assertRaises(Exception, self.agent.set_action_space, **spec)

    def test_agent_init(self):
        spec = {'a1': Integer(0, 1)}
        self.agent.init(self.task_spec)
        self.assertEqual(self.agent._action_space, Space(spec))
        self.assertEqual(self.agent.initialized, True)

    def test_agent_start_abstract(self):
        obs = {'o1': 1}
        action = self.agent.start(obs)
        self.assertEqual(action, None)

    def test_agent_start_overriden(self):
        def start(observation):
            return {'a1': 1}
        self.agent._start = start
        obs = {'o1': 1}
        action = self.agent.start(obs)
        self.assertEqual(action, {'a1': 1})

    def test_agent_step_abstract(self):
        reward = 0
        obs = {'o1': 1}
        action = self.agent.step(reward, obs)
        self.assertEqual(action, None)

    def test_agent_step_overriden(self):
        def step(reward, observation):
            return {'a1': 1}
        reward = 0
        obs = {'o1': 1}
        self.agent._step = step
        action = self.agent.step(reward, obs)
        self.assertEqual(action, {'a1': 1})

    def test_agent_end(self):
        self.assertEqual(self.agent.end(0), None)

    def test_agent_cleanup(self):
        self.assertEqual(self.agent.cleanup(), None)



if __name__ == '__main__':
    unittest.main()
