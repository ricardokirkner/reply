import unittest

from reply.mock.rlglue import RL_init, RL_start, RL_step, RL_cleanup
from reply.mock.rlglue import Reward_observation_action_terminal
from reply.mock.rlglue import RL_agent_message, RL_env_message
from reply.mock.rlglue import AgentLoader, EnvironmentLoader


class TestMockFunctions(unittest.TestCase):
    def test_rl_init(self):
        self.assertTrue(RL_init() is None)

    def test_rl_start(self):
        self.assertTrue(RL_start() is None)

    def test_rl_step(self):
        expected_roat = Reward_observation_action_terminal()
        roat = RL_step()
        self.assertEqual(roat.r, expected_roat.r)
        self.assertEqual(roat.terminal, expected_roat.terminal)
        self.assertTrue(roat.o.sameAs(expected_roat.o))
        self.assertTrue(roat.a.sameAs(expected_roat.a))

    def test_rl_cleanup(self):
        self.assertTrue(RL_cleanup() is None)

    def test_rl_agent_message(self):
        self.assertTrue(RL_agent_message('') is None)

    def test_rl_env_message(self):
        self.assertTrue(RL_env_message('') is None)


class TestMockClasses(unittest.TestCase):
    def test_loadAgent(self):
        loader = AgentLoader()
        self.assertTrue(loader.loadAgent(None) is None)

    def test_loadEnvironment(self):
        loader = EnvironmentLoader()
        self.assertTrue(loader.loadEnvironment(None) is None)


if __name__ == '__main__':
    unittest.main()
