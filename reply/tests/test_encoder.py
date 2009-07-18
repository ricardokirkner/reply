import unittest

from reply.datatypes import Double, Integer, Space
from reply.encoder import BucketEncoder, DummyEncoder, Encoder, SpaceEncoder
from reply.encoder import CompoundSpaceEncoder


class TestEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = Encoder()

    def test_encode(self):
        self.assertRaises(NotImplementedError, self.encoder.encode, None)

    def test_decode(self):
        self.assertRaises(NotImplementedError, self.encoder.decode, None)


class TestDummyEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = DummyEncoder()

    def test_encode(self):
        value = 1
        self.assertEqual(self.encoder.encode(value), value)

    def test_decode(self):
        value = 1
        self.assertEqual(self.encoder.decode(value), value)


class TestBucketEncoder(unittest.TestCase):
    def setUp(self):
        self.param = Double(0, 1)
        self.encoder = BucketEncoder(self.param)

    def test_encode(self):
        encoded_item = self.encoder.encode(self.param.min)
        self.assertEqual(encoded_item, 0)
        encoded_item = self.encoder.encode(self.param.max)
        self.assertEqual(encoded_item, 1)

        encoded_item = self.encoder.encode(0.49)
        self.assertEqual(encoded_item, 0)
        encoded_item = self.encoder.encode(0.5)
        self.assertEqual(encoded_item, 1)

    def test_encode_buckets(self):
        self.encoder.num_buckets = 10

        encoded_item = self.encoder.encode(self.param.min)
        self.assertEqual(encoded_item, 0)
        encoded_item = self.encoder.encode(self.param.max)
        self.assertEqual(encoded_item, 9)

        encoded_item = self.encoder.encode(0.3)
        self.assertEqual(encoded_item, 3)
        encoded_item = self.encoder.encode(0.73)
        self.assertEqual(encoded_item, 7)

        self.encoder.num_buckets = 5
        encoded_item = self.encoder.encode(0.27)
        self.assertEqual(encoded_item, 1)
        encoded_item = self.encoder.encode(0.19999999999)
        self.assertEqual(encoded_item, 0)

    def test_decode(self):
        item = self.encoder.decode(0)
        self.assertEqual(item, self.param.min)
        item = self.encoder.decode(1)
        self.assertEqual(item, 0.5)

        self.encoder.num_buckets = 10

        item = self.encoder.decode(3)
        self.assertEqual(item, 0.3)
        item = self.encoder.decode(7)
        self.assertEqual(item, 0.7)

        self.encoder.num_buckets = 4

        item = self.encoder.decode(2)
        self.assertEqual(item, 0.5)
        item = self.encoder.decode(3)
        self.assertEqual(item, 0.75)


class TestSpaceEncoder(unittest.TestCase):
    def setUp(self):
        space = Space({'choice': Integer(0, 9)})
        self.encoder = SpaceEncoder(space)

    def test_encode(self):
        item = {'choice': 5}
        encoded_item = self.encoder.encode(item)
        expected_encoded_item = (5,)
        self.assertEqual(encoded_item, expected_encoded_item)

    def test_decode(self):
        encoded_item = (5,)
        item = self.encoder.decode(encoded_item)
        expected_item = {'choice': 5}
        self.assertEqual(item, expected_item)


class TestCompoundSpaceEncoder(unittest.TestCase):
    def setUp(self):
        space = Space({'choice': Double(0, 1)})
        encoders = {'choice': BucketEncoder(space.choice)}
        self.encoder = CompoundSpaceEncoder(space, encoders)

    def test_encode(self):
        item = {'choice': 0.34}
        encoded_item = self.encoder.encode(item)
        expected_encoded_item = (0,)
        self.assertEqual(encoded_item, expected_encoded_item)
        item = {'choice': 0.82}
        encoded_item = self.encoder.encode(item)
        expected_encoded_item = (1,)
        self.assertEqual(encoded_item, expected_encoded_item)

    def test_decode(self):
        encoded_item = (0,)
        item = self.encoder.decode(encoded_item)
        expected_item = {'choice': 0.0}
        self.assertEqual(item, expected_item)
        encoded_item = (1,)
        item = self.encoder.decode(encoded_item)
        expected_item = {'choice': 0.5}
        self.assertEqual(item, expected_item)


if __name__ == '__main__':
    unittest.main()
