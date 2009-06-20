import unittest

from reply.agent import Agent
from reply.datatypes import Char, Double, Integer, Space
from reply.util import TaskSpec


class TestAgent(unittest.TestCase):

    def setUp(self):
        self.agent = Agent()
        self.task_spec_str = "VERSION RL-Glue-3.0 PROBLEMTYPE episodic " \
            "DISCOUNTFACTOR 1.0" \
            "OBSERVATIONS INTS (0 1) ACTIONS INTS (0 1) REWARDS (0 1) " \
            "EXTRA OBSERVATIONS INTS o1 ACTIONS INTS a1"
        self.task_spec = TaskSpec.parse(self.task_spec_str)

    def test_agent_builder(self):
        self.assertEqual(self.agent.initialized, False)

    def test_agent_set_action_space_already_initialized(self):
        self.agent.init(self.task_spec)
        spec = {'a1': Integer(0, 1)}
        self.assertRaises(Exception, self.agent.set_action_space, **spec)

    def test_agent_set_action_space_with_space(self):
        space = Space({'a1': Integer(0, 3)})
        self.agent.set_action_space(space)
        self.assertEqual(self.agent._action_space, space)

    def test_agent_set_action_space_without_space(self):
        space = Space()
        self.agent.set_action_space()
        self.assertEqual(self.agent._action_space, space)

    def test_agent_set_action_space_with_kwargs(self):
        spec = {'a1': Integer(0, 3)}
        space = Space(spec)
        self.agent.set_action_space(**spec)
        self.assertEqual(self.agent._action_space, space)

    def test_agent_set_observation_space_already_initialized(self):
        self.agent.init(self.task_spec)
        spec = {'o1': Integer(0, 1)}
        self.assertRaises(Exception, self.agent.set_observation_space, **spec)

    def test_agent_set_observation_space_with_space(self):
        space = Space({'o1': Integer(0, 3)})
        self.agent.set_observation_space(space)
        self.assertEqual(self.agent._observation_space, space)

    def test_agent_set_observation_space_without_space(self):
        space = Space()
        self.agent.set_observation_space()
        self.assertEqual(self.agent._observation_space, space)

    def test_agent_set_observation_space_with_kwargs(self):
        spec = {'o1': Integer(0, 3)}
        space = Space(spec)
        self.agent.set_observation_space(**spec)
        self.assertEqual(self.agent._observation_space, space)

    def test_agent_init(self):
        action_spec = {'a1': Integer(0, 1), '': []}
        observation_spec = {'o1': Integer(0, 1), '': []}
        self.agent.init(self.task_spec)
        self.assertEqual(self.agent._action_space, Space(action_spec))
        self.assertEqual(self.agent._observation_space, Space(observation_spec))
        self.assertEqual(self.agent.initialized, True)

    def test_agent_start_abstract(self):
        obs = {'o1': 1}
        action = self.agent.start(obs)
        self.assertEqual(action, {})

    def test_agent_step_abstract(self):
        reward = 0
        obs = {'o1': 1}
        action = self.agent.step(reward, obs)
        self.assertEqual(action, {})

    def test_agent_end(self):
        self.assertEqual(self.agent.end(0), None)

    def test_agent_cleanup(self):
        self.assertEqual(self.agent.cleanup(), None)



if __name__ == '__main__':
    unittest.main()
