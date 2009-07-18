import unittest

from reply.datatypes import Integer, Space, Model
from reply.encoder import SpaceEncoder
from reply.learner import Learner, QLearner, SarsaLearner
from reply.policy import EGreedyPolicy
from reply.storage import TableStorage


class TestLearner(unittest.TestCase):
    def setUp(self):
        self.policy = None
        self.learner = Learner(self)

    def test_learner_builder(self):
        self.assertEqual(self.learner.policy, self.policy)

    def test_learner_update(self):
        self.assertRaises(NotImplementedError, self.learner.update,
                          None, None, None, None)


class TestQLearner(unittest.TestCase):
    def setUp(self):
        self.observations = Space({'o': Integer(0, 1)})
        self.actions = Space({'a': Integer(0, 1)})
        self.model = Model(self.observations, self.actions)
        self.storage = TableStorage(self)
        self.policy = EGreedyPolicy(self)
        self.learning_rate = 1.0
        self.learning_rate_decay = 0.9
        self.learning_rate_min = 0.1
        self.value_discount = 0.9
        self.learner = QLearner(self)

    def test_learner_builder(self):
        self.assertEqual(self.learner.policy, self.policy)
        self.assertEqual(self.learner.learning_rate, self.learning_rate)
        self.assertEqual(self.learner.learning_rate_decay,
                         self.learning_rate_decay)
        self.assertEqual(self.learner.learning_rate_min, self.learning_rate_min)
        self.assertEqual(self.learner.value_discount, self.value_discount)

    def test_learner_update(self):
        state = {'o': 0}
        next_state = {'o': 1}
        action = {'a': 1}
        reward = 1

        self.learner.update(state, action, reward, next_state)
        state_value = self.learner.policy.storage.get((state, action))
        expected_state_value = 1
        self.assertEqual(state_value, expected_state_value)

    def test_learner_update_end(self):
        state = {'o': 1}
        action = {'a': 1}
        reward = 0

        self.learner.update(state, action, reward, None)
        state_value = self.learner.policy.storage.get((state, action))
        expected_state_value = 0
        self.assertEqual(state_value, expected_state_value)


class TestSarsaLearner(unittest.TestCase):
    def setUp(self):
        self.observations = Space({'o': Integer(0, 1)})
        self.actions = Space({'a': Integer(0, 1)})
        self.model = Model(self.observations, self.actions)
        self.storage = TableStorage(self)
        self.policy = EGreedyPolicy(self)
        self.learning_rate = 1.0
        self.learning_rate_decay = 0.9
        self.learning_rate_min = 0.1
        self.value_discount = 0.9
        self.learner = SarsaLearner(self)

    def test_learner_builder(self):
        self.assertEqual(self.learner.policy, self.policy)
        self.assertEqual(self.learner.learning_rate, self.learning_rate)
        self.assertEqual(self.learner.learning_rate_decay,
                         self.learning_rate_decay)
        self.assertEqual(self.learner.learning_rate_min, self.learning_rate_min)
        self.assertEqual(self.learner.value_discount, self.value_discount)

    def test_learner_update(self):
        state = {'o': 0}
        next_state = {'o': 1}
        action = {'a': 1}
        reward = 1

        self.learner.update(state, action, reward, next_state)
        state_value = self.learner.policy.storage.get((state, action))
        expected_state_value = 1
        self.assertEqual(state_value, expected_state_value)

    def test_learner_update_end(self):
        state = {'o': 1}
        action = {'a': 1}
        reward = 0

        self.learner.update(state, action, reward, None)
        state_value = self.learner.policy.storage.get((state, action))
        expected_state_value = 0
        self.assertEqual(state_value, expected_state_value)


if __name__ == '__main__':
    unittest.main()
