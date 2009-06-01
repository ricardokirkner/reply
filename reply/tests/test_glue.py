import simplejson
import unittest

from rlglue.types import Action, Observation, Reward_observation_terminal

from reply.agent import Agent
from reply.environment import Environment
from reply.experiment import Experiment
from reply.glue import adapt, RlGlueProxyAgent, RlGlueProxyEnvironment
from reply.glue import RlGlueProxyExperiment
from reply.types import Char, Double, Integer, Space


class TestRLGlue(unittest.TestCase):

    def test_adapt_to_action(self):
        space = Space(dict(choice=Integer(0, 1)))
        reply_action = dict(choice=0)
        action = adapt(reply_action, space, Action)

        expected_action = Action()
        expected_action.intArray = [0]

        self.assertTrue(action.sameAs(expected_action))

    def test_adapt_from_action(self):
        space = Space(dict(choice=Integer(0, 1)))
        action = Action()
        action.intArray = [0]
        reply_action = adapt(action, space)

        expected_action = dict(choice=0)

        self.assertEquals(reply_action, expected_action)


class TestRLGlueAgent(unittest.TestCase):

    def setUp(self):
        self.agent = Agent()
        self.proxy = RlGlueProxyAgent(self.agent)
        self.task_spec = "VERSION RL-Glue-3.0 PROBLEMTYPE episodic DISCOUNTFACTOR 1.0 " \
            "OBSERVATIONS INTS (-10 10) (-10 10) ACTIONS INTS (0 3) REWARDS (-1 0) " \
            "EXTRA OBSERVATIONS INTS x y ACTIONS INTS dir"

    def test_builder(self):
        self.assertEqual(self.proxy.agent, self.agent)

    def test_agent_init(self):
        self.assertEqual(self.proxy.agent_init(self.task_spec), None)

    def test_agent_start(self):
        self.proxy.agent_init(self.task_spec)
        observation = Observation()
        observation.intArray = [2, 2]
        action = self.proxy.agent_start(observation)

        expected_action = Action()
        self.assertTrue(action.sameAs(expected_action))

    def test_agent_step(self):
        self.proxy.agent_init(self.task_spec)
        observation = Observation()
        observation.intArray = [1,0]
        reward = -1
        action = self.proxy.agent_step(reward, observation)

        expected_action = Action()
        self.assertTrue(action.sameAs(expected_action))

    def test_agent_end(self):
        reward = 0
        self.assertEqual(self.proxy.agent_end(0), None)

    def test_agent_cleanup(self):
        self.assertEqual(self.proxy.agent_cleanup(), None)

    def test_agent_message(self):
        def greet(who='World'):
            return "Hello, %s!" % who
        self.agent.on_greet = greet
        message = {'function_name': 'greet', 'args': ('Agent',), 'kwargs': {}}
        response = self.proxy.agent_message(simplejson.dumps(message))
        self.assertEqual(simplejson.loads(response), "Hello, Agent!")


class TestRLGlueEnvironment(unittest.TestCase):

    def setUp(self):
        class TestEnvironment(Environment):
            problem_type = 'episodic'
            discount_factor = 1.0
            observations_spec = {'x': Integer(-10, 10), 'y': Integer(-10, 10)}
            actions_spec = {'dir': Integer(0, 3)}
            rewards = Integer(-1, 0)

        self.environment = TestEnvironment()
        self.proxy = RlGlueProxyEnvironment(self.environment)
        self.task_spec = "VERSION RL-Glue-3.0 PROBLEMTYPE episodic DISCOUNTFACTOR 1.0 " \
            "OBSERVATIONS INTS (-10 10) (-10 10) ACTIONS INTS (0 3) REWARDS (-1 0) " \
            "EXTRA OBSERVATIONS INTS x y ACTIONS INTS dir"

    def test_builder(self):
        self.assertEqual(self.proxy.environment, self.environment)

    def test_env_init(self):
        self.assertEqual(self.proxy.env_init(), self.task_spec)

    def test_env_start(self):
        expected_observation = Observation()
        observation = self.proxy.env_start()
        self.assertTrue(observation.sameAs(expected_observation))

    def test_env_step(self):
        expected_observation = Observation()
        action = Action()
        rot = self.proxy.env_step(action)
        self.assertEqual(rot.r, 0.0)
        self.assertTrue(rot.o.sameAs(expected_observation))
        self.assertEqual(rot.terminal, False)

    def test_env_cleanup(self):
        self.assertEqual(self.proxy.env_cleanup(), None)

    def test_env_message(self):
        def greet(who='World'):
            return "Hello, %s!" % who
        self.environment.on_greet = greet
        message = {'function_name': 'greet', 'args': ('Environment',), 'kwargs': {}}
        response = self.proxy.env_message(simplejson.dumps(message))
        self.assertEqual(simplejson.loads(response), "Hello, Environment!")


class TestRLGlueExperiment(unittest.TestCase):

    def _test_builder(self):
        pass

    def _test_init(self):
        pass

    def _test_start(self):
        pass

    def _test_step(self):
        pass

    def _test_cleanup(self):
        pass

    def _test_run(self):
        pass



if __name__ == '__main__':
    unittest.main()
