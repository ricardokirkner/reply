import simplejson
import unittest

from rlglue.types import Action, Observation, Reward_observation_action_terminal

import reply.glue
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

    def test_adapt_to_complex_action(self):
        space = Space({'x': Double(-1.0, 1.0),
                       'y': Double(-1.0, 1.0),
                       'angle': Integer(0, 359),
                       'spin': Char()},
                       {Integer: ['angle'],
                        Double: ['x', 'y'],
                        Char: ['spin']})
        reply_action = {'x': -0.23, 'y': 0.02,
                        'angle': 23, 'spin': 'u'}
        action = adapt(reply_action, space, Action)

        expected_action = Action()
        expected_action.intArray = [23]
        expected_action.doubleArray = [-0.23, 0.02]
        expected_action.charArray = ['u']

        self.assertTrue(action.sameAs(expected_action))

    def test_adapt_from_complex_action(self):
        space = Space({'x': Double(-1.0, 1.0),
                       'y': Double(-1.0, 1.0),
                       'angle': Integer(0, 359),
                       'spin': Char()},
                       {Integer: ['angle'],
                        Double: ['x', 'y'],
                        Char: ['spin']})
        action = Action()
        action.intArray = [23]
        action.doubleArray = [-0.23, 0.02]
        action.charArray = ['u']
        reply_action = adapt(action, space)

        expected_action = {'x': -0.23, 'y': 0.02,
                           'angle': 23, 'spin': 'u'}

        self.assertEqual(reply_action, expected_action)


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

    def RL_init(self):
        pass

    def RL_start(self):
        pass

    def RL_step(self):
        terminal = False
        reward = 0
        observation = Observation()
        action = Action()
        roat = Reward_observation_action_terminal()
        roat.r = reward
        roat.o = observation
        roat.a = action
        roat.terminal = terminal

        return roat

    def RL_cleanup(self):
        pass

    def setUp(self):
        self.experiment = Experiment()
        self.proxy = RlGlueProxyExperiment(self.experiment)
        # mock rlglue methods
        reply.glue.RL_init = self.RL_init
        reply.glue.RL_start = self.RL_start
        reply.glue.RL_step = self.RL_step
        reply.glue.RL_cleanup = self.RL_cleanup

    def test_builder(self):
        self.assertEqual(self.proxy.experiment, self.experiment)
        self.assertEqual(self.proxy._initialized, False)
        self.assertEqual(self.proxy._started, False)

    def test_init(self):
        self.assertEqual(self.proxy.init(), None)

    def test_start(self):
        self.assertEqual(self.proxy.start(), None)

    def test_step(self):
        roat = self.proxy.step()
        self.assertEqual(roat.terminal, False)
        self.assertEqual(roat.r, 0)
        self.assertTrue(roat.o.sameAs(Observation()))
        self.assertTrue(roat.a.sameAs(Action()))

    def test_cleanup(self):
        self.assertEqual(self.proxy.cleanup(), None)

    def test_run(self):
        self.assertEqual(self.proxy.run(), None)



if __name__ == '__main__':
    unittest.main()
