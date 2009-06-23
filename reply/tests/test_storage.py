import numpy
import unittest

from reply.datatypes import Space, Integer
from reply.encoder import SpaceEncoder, StateActionEncoder
from reply.storage import Storage, TableStorage

identity = lambda x: True

class TestStorage(unittest.TestCase):
    def setUp(self):
        self.storage = Storage()

    def test_storage_getitem(self):
        self.assertRaises(NotImplementedError, self.storage.__getitem__, 1)

    def test_storage_setitem(self):
        self.assertRaises(NotImplementedError, self.storage.__setitem__, 1, 1)

    def test_storage_get(self):
        self.assertRaises(NotImplementedError, self.storage.get, 1)

    def test_storage_set(self):
        self.assertRaises(NotImplementedError, self.storage.set, 1, 1)

    def test_storage_clear(self):
        self.assertRaises(NotImplementedError, self.storage.clear)

    def test_storage_filter(self):
        self.assertRaises(NotImplementedError, self.storage.filter, 1, identity)


class TestTableStorage(unittest.TestCase):
    def setUp(self):
        observations = Space({'o': Integer(0, 0)})
        actions = Space({'a': Integer(0, 0)})
        state_encoder = SpaceEncoder(observations)
        action_encoder = SpaceEncoder(actions)
        self.size = (observations.size, actions.size)
        self.data = numpy.zeros(self.size)
        self.storage = TableStorage(state_encoder, action_encoder)

    def test_storage_init_default(self):
        self.assertTrue(isinstance(self.storage.encoder, StateActionEncoder))
        self.assertTrue((self.storage.data == self.data).all())

    def test_storage_get(self):
        observation = {'o': 0}
        action = {'a': 0}
        value = self.storage.get((observation, action))
        self.assertEqual(value, 0.0)
        self.assertRaises(IndexError, self.storage.get, (observation, {'a': 1}))

    def test_storage_set(self):
        observation = {'o': 0}
        action = {'a': 0}
        self.assertEqual(self.storage.get((observation, action)), 0.0)
        self.storage.set((observation, action), 5)
        self.assertEquals(self.storage.get((observation, action)), 5.0)

    def test_storage_clear(self):
        observation = {'o': 0}
        action = {'a': 0}
        self.assertEqual(self.storage.get((observation, action)), 0.0)
        self.storage.set((observation, action), 1)
        self.assertEqual(self.storage.get((observation, action)), 1.0)
        self.storage.clear()
        self.assertEqual(self.storage.get((observation, action)), 0.0)
        self.assertTrue((self.storage.data == self.data).all())

    def test_storage_filter(self):
        observations = Space({'o': Integer(0, 1)})
        actions = Space({'a': Integer(0, 5)})
        observation_encoder = SpaceEncoder(observations)
        action_encoder = SpaceEncoder(actions)
        storage = TableStorage(observation_encoder, action_encoder)
        observation = {'o': 0}
        for action_value in range(6):
            action = {'a': action_value}
            storage.set((observation, action), action_value)
        value = storage.filter((observation,), max)
        self.assertEqual(value, 5.0)

    def test_storage_filter_many(self):
        observations = Space({'o': Integer(0, 1)})
        actions = Space({'a': Integer(0, 5)})
        observation_encoder = SpaceEncoder(observations)
        action_encoder = SpaceEncoder(actions)
        storage = TableStorage(observation_encoder, action_encoder)
        observation = {'o': 0}
        for action_value in range(6):
            action = {'a': action_value}
            storage.set((observation, action), action_value)
        values = storage.filter((observation,), lambda x: filter(lambda y: y % 2 == 0, x))
        self.assertEqual(values, [0, 2, 4])


if __name__ == '__main__':
    unittest.main()
