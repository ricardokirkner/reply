import numpy
import unittest

from reply.datatypes import Space, Integer, Model, Double
from reply.storage import Storage, TableStorage, BucketStorage


class TestStorage(unittest.TestCase):
    def setUp(self):
        observations = Space({'o': Integer(0, 0)})
        actions = Space({'a': Integer(0, 0)})
        self.model = Model(observations, actions)
        self.storage = Storage(self)

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


class TestTableStorage(unittest.TestCase):
    def setUp(self):
        observations = Space({'o': Integer(0, 0)})
        actions = Space({'a': Integer(0, 0)})
        shape = (1, 1)
        self.data = numpy.zeros(shape)
        self.model = Model(observations, actions)
        self.storage = TableStorage(self)

    def test_storage_init_default(self):
        self.assertTrue((self.storage.data == self.data).all())

    def test_storage_get(self):
        observation = {'o': 0}
        action = {'a': 0}
        value = self.storage.get(observation, action)
        self.assertEqual(value, 0.0)
        self.assertRaises(ValueError, self.storage.get, observation, {'a': 1})

    def test_storage_set(self):
        observation = {'o': 0}
        action = {'a': 0}
        self.assertEqual(self.storage.get(observation, action), 0.0)
        self.storage.set(observation, action, 5)
        self.assertEquals(self.storage.get(observation, action), 5.0)

    def test_storage_clear(self):
        observation = {'o': 0}
        action = {'a': 0}
        self.assertEqual(self.storage.get(observation, action), 0.0)
        self.storage.set(observation, action, 1)
        self.assertEqual(self.storage.get(observation, action), 1.0)
        self.storage.clear()
        self.assertEqual(self.storage.get(observation, action), 0.0)
        self.assertTrue((self.storage.data == self.data).all())

    def test_storage_get_max_value(self):
        observations = Space({'o': Integer(0, 1)})
        actions = Space({'a': Integer(0, 5)})

        self.model = Model(observations, actions)
        storage = TableStorage(self)

        observation = {'o': 0}
        for action_value in range(6):
            action = {'a': action_value}
            storage.set(observation, action, action_value)
        value = storage.get_max_value(observation)
        self.assertEqual(value, 5.0)

    def test_get_observations(self):
        observations = Space({'o1': Integer(0, 1),
                              'o2': Integer(2, 4)})
        actions = Space({'a1': Integer(0, 1),
                         'a2': Integer(1, 2)})
        self.model = Model(observations, actions)
        storage = TableStorage(self)
        observations = list(storage.get_observations())
        expected_observations = [{'o1': 0, 'o2': 2}, {'o1': 0, 'o2': 3},
                                 {'o1': 0, 'o2': 4}, {'o1': 1, 'o2': 2},
                                 {'o1': 1, 'o2': 3}, {'o1': 1, 'o2': 4}]
        self.assertEqual(observations, expected_observations)

    def test_get_actions(self):
        observations = Space({'o1': Integer(0, 1),
                              'o2': Integer(2, 4)})
        actions = Space({'a1': Integer(0, 1),
                         'a2': Integer(1, 2)})
        self.model = Model(observations, actions)
        storage = TableStorage(self)
        actions = list(storage.get_actions())
        expected_actions = [{'a1': 0, 'a2': 1}, {'a1': 0, 'a2': 2},
                            {'a1': 1, 'a2': 1}, {'a1': 1, 'a2': 2}]
        self.assertEqual(actions, expected_actions)

    def test_get_action(self):
        observations = Space({'o1': Integer(0, 1),
                              'o2': Integer(2, 4)})
        actions = Space({'a1': Integer(0, 1),
                         'a2': Integer(1, 2)})

        self.model = Model(observations, actions)
        storage = TableStorage(self)

        action = storage.get_action((0,1))
        expected_action = {'a1': 0, 'a2': 2}
        self.assertEqual(action, expected_action)

class TestBucketStorage(unittest.TestCase):
    def test_simple_obs(self):
        observations = Space({'o': Double(0, 1)})
        actions = Space({'a': Double(0, 1)})
        self.storage_observation_buckets = {'o':10}
        self.storage_action_buckets = {'o':10}
        self.model = Model(observations, actions)
        self.storage = BucketStorage(self)

        for i in range(10):
            self.assertEqual((i,) , self.storage.encode({'o':i/10.0}))

    def test_simple_obs_with_offset(self):
        observations = Space({'o': Double(-1, 1)})
        actions = Space({'a': Double(0, 1)})
        self.storage_observation_buckets = {'o':10}
        self.storage_action_buckets = {'o':10}
        self.model = Model(observations, actions)
        self.storage = BucketStorage(self)

        for i in range(10):
            self.assertEqual((i,) , self.storage.encode({'o':(i-4.9)/5}))

    def test_max_value(self):
        observations = Space({'o': Double(0, 1)})
        actions = Space({'a': Double(0, 1)})
        self.storage_observation_buckets = {'o':10}
        self.storage_action_buckets = {'o':10}
        self.model = Model(observations, actions)
        self.storage = BucketStorage(self)

        for i in range(10):
            self.storage.set(
                {'o':(i/10.0)},
                {'a':(i/10.0)},
                i)
        for i in range(10):
            self.assertEqual(i ,
                self.storage.get_max_value({'o':(i/10.0)}))

    def test_get_actions(self):
        observations = Space({'o': Double(0, 1)})
        actions = Space({'a': Double(0, 1)})
        self.storage_observation_buckets = {'o':10}
        self.storage_action_buckets = {'a':10}
        self.model = Model(observations, actions)
        self.storage = BucketStorage(self)

        for i, a in enumerate(self.storage.get_actions()):
            self.assertEqual(i/10.0, a['a'])

    def test_get_action(self):
        observations = Space({'o': Double(0, 1)})
        actions = Space({'a': Double(0, 1)})
        self.storage_observation_buckets = {'o':10}
        self.storage_action_buckets = {'a':10}
        self.model = Model(observations, actions)
        self.storage = BucketStorage(self)

        for i, a in enumerate(self.storage.get_actions()):
            self.assertEqual(self.storage.get_action((i,)), a)

    def test_get_actions_encode(self):
        observations = Space({'o': Double(0, 1)})
        actions = Space({'a': Double(-1, 1)})
        self.storage_observation_buckets = {'o':10}
        self.storage_action_buckets = {'a':20}
        self.model = Model(observations, actions)
        self.storage = BucketStorage(self)

        for i, a in enumerate(self.storage.get_actions()):
            self.assertEqual( self.storage.encode({'o':0}, a), (0,i) )

if __name__ == '__main__':
    unittest.main()
