import unittest

from reply.datatypes import Integer, Space
from reply.encoder import Encoder, DummyEncoder, SpaceEncoder, StateActionEncoder


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


class TestStateActionEncoder(unittest.TestCase):
    def setUp(self):
        state_space = Space({'state': Integer(0, 0)})
        action_space = Space({'choice': Integer(0, 9)})
        state_encoder = SpaceEncoder(state_space)
        action_encoder = SpaceEncoder(action_space)
        self.encoder = StateActionEncoder(state_encoder, action_encoder)

    def test_encode(self):
        item = ({'state': 0}, {'choice': 3})
        encoded_item = self.encoder.encode(item)
        expected_encoded_item = (0, 3)
        self.assertEqual(encoded_item, expected_encoded_item)
        item = ({'state': 0},)
        encoded_item = self.encoder.encode(item)
        expected_encoded_item = (0,)
        self.assertEqual(encoded_item, expected_encoded_item)
        item = {'state': 0}
        encoded_item = self.encoder.encode(item)
        expected_encoded_item = (0,)
        self.assertEqual(encoded_item, expected_encoded_item)

    def test_encode_invalid_item(self):
        item = ({'state': 0}, {'choice': 3}, {'other': 0})
        self.assertRaises(ValueError, self.encoder.encode, item)

    def test_decode(self):
        encoded_item = (0, 3)
        item = self.encoder.decode(encoded_item)
        expected_item = ({'state': 0}, {'choice': 3})
        self.assertEqual(item, expected_item)
        encoded_item = (0,)
        item = self.encoder.decode(encoded_item)
        expected_item = ({'state': 0},)
        self.assertEqual(item, expected_item)
        encoded_item = 0
        item = self.encoder.decode(encoded_item)
        expected_item = ({'state': 0},)
        self.assertEqual(item, expected_item)

    def test_decode_invalid_item(self):
        encoded_item = (0, 3, 0)
        self.assertRaises(ValueError, self.encoder.decode, encoded_item)

if __name__ == '__main__':
    unittest.main()
