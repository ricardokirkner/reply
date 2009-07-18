import unittest

from reply.datatypes import Integer, Space, Model
from reply.encoder import SpaceEncoder
from reply.policy import Policy, EGreedyPolicy, SoftMaxPolicy
from reply.storage import TableStorage


class TestPolicy(unittest.TestCase):
    def setUp(self):
        self.storage = None
        self.policy = Policy(self)

    def test_builder(self):
        self.assertTrue(self.policy.storage is None)

    def test_equal(self):
        policy2 = Policy(self)
        self.assertEqual(self.policy, policy2)

    def test_select_action(self):
        self.assertRaises(NotImplementedError, self.policy.select_action, None)


class TestEGreedyPolicy(unittest.TestCase):
    def setUp(self):
        self.observations = Space({'o': Integer(0, 1)})
        self.actions = Space({'a': Integer(0, 1)})
        self.model = Model(self.observations, self.actions)
        self.storage = TableStorage(self)
        self.storage.set(({'o': 0}, {'a': 1}), 1)
        self.policy = EGreedyPolicy(self)

    def test_builder(self):
        self.assertEqual(self.policy.storage, self.storage)
        self.assertEqual(self.policy.random_action_rate, 0)
        self.assertEqual(self.policy.random_action_rate_decay, 1)
        self.assertEqual(self.policy.random_action_rate_min, 0)

    def test_equal(self):
        policy2 = EGreedyPolicy(self)
        self.assertEqual(self.policy, policy2)

    def test_select_action(self):
        observation = {'o': 0}
        expected_action = {'a': 1}

        action = self.policy.select_action(observation)
        self.assertEqual(action, expected_action)

    def test_get_mappings(self):
        expected_mappings = [({'o': 0}, {'a': 1}), ({'o': 1}, {'a': 0})]
        mappings = self.policy.get_mappings()
        self.assertEqual(mappings, expected_mappings)


class TestSoftMaxPolicy(unittest.TestCase):
    def setUp(self):
        self.observations = Space({'o': Integer(0, 1)})
        self.temperature = 0
        self.actions = Space({'a': Integer(0, 1)})
        self.model = Model(self.observations, self.actions)
        self.storage = TableStorage(self)
        self.storage.set(({'o': 0}, {'a': 1}), 1)
        self.policy = SoftMaxPolicy(self)

    def test_builder(self):
        self.assertEqual(self.policy.storage, self.storage)
        self.assertEqual(self.policy.temperature, 0)

    def test_equal(self):
        policy2 = SoftMaxPolicy(self)
        old = self.temperature
        self.temperature = 1
        policy3 = SoftMaxPolicy(self)
        self.temperature = old
        self.assertEqual(self.policy, policy2)
        self.assertNotEqual(self.policy, policy3)

    def test_select_action(self):
        observation = {'o': 0}
        expected_action = {'a': 1}

        action = self.policy.select_action(observation)
        self.assertEqual(action, expected_action)

    def test_select_action_temperature(self):
        old = self.temperature
        self.temperature = 0.01

        policy = SoftMaxPolicy(self)
        observation = {'o': 0}
        expected_action = {'a': 1}
        num_hits = 0
        num_tries = 100
        for i in range(num_tries):
            action = policy.select_action(observation)
            if action == expected_action:
                num_hits += 1
        # num_hits might occasionally not be exactly num_tries
        self.assertTrue(1.0 - num_hits/float(num_tries) < 1.0/num_tries)

    def test_get_mappings(self):
        expected_mappings = [({'o': 0}, {'a': 1}), ({'o': 1}, {'a': 0})]
        mappings = self.policy.get_mappings()
        self.assertEqual(mappings, expected_mappings)


if __name__ == '__main__':
    unittest.main()
