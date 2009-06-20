import numpy
import unittest

from reply.encoder import DummyEncoder
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
        self.size = (1, 1)
        self.data = numpy.zeros(self.size)
        self.storage = TableStorage(self.size)

    def test_storage_init_default(self):
        self.assertTrue(isinstance(self.storage.encoder, DummyEncoder))
        self.assertEqual(self.storage.data, self.data)

    def test_storage_init_encoder(self):
        encoder = DummyEncoder()
        storage = TableStorage(self.size, encoder)
        self.assertEqual(storage.encoder, encoder)
        self.assertEqual(storage.data, self.data)

    def test_storage_get(self):
        value = self.storage.get((0, 0))
        self.assertEqual(value, 0.0)
        self.assertRaises(IndexError, self.storage.get, (0, 1))

    def test_storage_set(self):
        self.assertEqual(self.storage.get((0, 0)), 0.0)
        self.storage.set((0, 0), 5)
        self.assertEquals(self.storage.get((0, 0)), 5.0)

    def test_storage_clear(self):
        self.assertEqual(self.storage.get((0, 0)), 0.0)
        self.storage.set((0, 0), 1)
        self.assertEqual(self.storage.get((0, 0)), 1.0)
        self.storage.clear()
        self.assertEqual(self.storage.get((0, 0)), 0.0)
        self.assertEqual(self.storage.data, self.data)

    def test_storage_filter(self):
        storage = TableStorage((1, 10))
        for item in range(10):
            storage.set((0, item), item)
        value = storage.filter((0,), max)
        self.assertEqual(value, 9.0)

    def test_storage_filter_many(self):
        storage = TableStorage((1, 10))
        for item in range(10):
            storage.set((0, item), item)
        values = storage.filter((0,), lambda x: filter(lambda y: y % 2 == 0, x))
        self.assertEqual(values, [0, 2, 4, 6, 8])


if __name__ == '__main__':
    unittest.main()
