import unittest

from reply.agent import Agent, LearningAgent
from reply.storage import TableStorage
from reply.policy import EGreedyPolicy
from reply.learner import QLearner
from reply.datatypes import Integer, Model, Space
from reply.util import TaskSpec


class TestAgent(unittest.TestCase):

    def setUp(self):
        self.agent = Agent(policy_class=EGreedyPolicy,
                           storage_class=TableStorage)
        self.task_spec_str = "VERSION RL-Glue-3.0 PROBLEMTYPE episodic " \
            "DISCOUNTFACTOR 1.0 " \
            "OBSERVATIONS INTS (0 1) ACTIONS INTS (0 1) REWARDS (0 1) " \
            "EXTRA OBSERVATIONS INTS o1 ACTIONS INTS a1"
        self.task_spec = TaskSpec.parse(self.task_spec_str)
        actions = Space({'a1': Integer(0, 1)})
        observations = Space({'o1': Integer(0, 1)})
        self.model = Model(observations, actions)

    def test_agent_builder(self):
        self.assertEqual(self.agent.initialized, False)

    def test_agent_init(self):
        action_spec = {'a1': Integer(0, 1)}
        observation_spec = {'o1': Integer(0, 1)}
        self.agent.init(self.task_spec)
        self.assertEqual(self.agent.model.actions, Space(action_spec))
        self.assertEqual(self.agent.model.observations, Space(observation_spec))
        self.assertEqual(self.agent.initialized, True)

    def test_agent_init_model(self):
        self.agent.model = self.model
        self.agent.init('')
        self.assertEqual(self.agent.model, self.model)
        self.assertEqual(self.agent.initialized, True)

    def test_agent_start_abstract(self):
        obs = {'o1': 1}
        self.agent.model = self.model
        self.agent.init('')
        action = self.agent.start(obs)
        self.assertEqual(action, {'a1': 0})

    def test_agent_step_abstract(self):
        reward = 0
        obs = {'o1': 1}
        self.agent.model = self.model
        self.agent.init('')
        action = self.agent.step(reward, obs)
        self.assertEqual(action, {'a1': 0})

    def test_agent_end(self):
        self.assertEqual(self.agent.end(0), None)

    def test_agent_cleanup(self):
        self.assertEqual(self.agent.cleanup(), None)


class TestLearningAgent(unittest.TestCase):
    def setUp(self):
        class DummyLearningAgent(LearningAgent):
            storage_class = TableStorage
            policy_class = EGreedyPolicy
            learner_class = QLearner
            observations = Space({'o1': Integer(0, 1)})
            actions = Space({'a1': Integer(0, 1)})
            model = Model({'spec': {'o1': Integer(0, 1)}},
                          {'spec': {'a1': Integer(0, 1)}})
            learning_rate = 0
            learning_rate_decay = 0

        self.agent = DummyLearningAgent()
        self.task_spec_str = "VERSION RL-Glue-3.0 PROBLEMTYPE episodic " \
            "DISCOUNTFACTOR 1.0" \
            "OBSERVATIONS INTS (0 1) ACTIONS INTS (0 1) REWARDS (0 1) " \
            "EXTRA OBSERVATIONS INTS o1 ACTIONS INTS a1"
        self.task_spec = TaskSpec.parse(self.task_spec_str)

    def test_init(self):
        self.agent.init(self.task_spec)
        self.observations = Space({'o1': Integer(0, 1)})
        self.actions = Space({'a1': Integer(0, 1)})
        self.model = Model(self.observations, self.actions)
        self.storage = TableStorage(self)
        policy = EGreedyPolicy(self)
        learner = QLearner(self.agent)
        self.assertEqual(self.agent.learner, learner)
        self.assertEqual(self.agent.last_observation, None)
        self.assertEqual(self.agent.last_action, None)

    def test_start(self):
        observation = {'o1': 0}
        expected_action = {'a1': 0}
        self.agent.init(self.task_spec)
        action = self.agent.start(observation)
        self.assertEqual(action, expected_action)

    def test_step(self):
        reward = 0
        observation = {'o1': 0}
        expected_action = {'a1': 0}
        self.agent.init(self.task_spec)
        self.agent.start(observation)
        action = self.agent.step(reward, observation)
        self.assertEqual(action, expected_action)

    def test_end(self):
        reward = 0
        observation = {'o1': 0}
        self.agent.init(self.task_spec)
        self.agent.start(observation)
        self.assertEqual(self.agent.end(reward), None)


if __name__ == '__main__':
    unittest.main()
