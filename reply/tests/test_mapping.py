import unittest

from reply.datatypes import Integer, Space
from reply.mapping import IdentityMapping, Mapping, OffsetIdentityMapping


class TestMapping(unittest.TestCase):
    def setUp(self):
        self.domain = Space({'o': Integer(2, 3)})
        self.image = Space({'o': Integer(0, 1)})
        self.mapping = Mapping(self.domain, self.image)

    def test_builder(self):
        self.assertEqual(self.mapping.domain, self.domain)
        self.assertEqual(self.mapping.image, self.image)

    def test_value(self):
        self.assertRaises(NotImplementedError, self.mapping.value, None)
        self.assertRaises(NotImplementedError,
                          lambda x: self.mapping.value(x, inverse=True), None)


class TestIdentityMapping(unittest.TestCase):
    def setUp(self):
        self.domain = Space({'o': Integer(2, 3)})
        self.mapping = IdentityMapping(self.domain)

    def test_builder(self):
        self.assertEqual(self.mapping.domain, self.domain)
        self.assertEqual(self.mapping.image, self.domain)

    def test_value(self):
        value = {'o': 2}
        value_image = (2,)
        self.assertEqual(self.mapping.value(value), value_image)
        self.assertEqual(self.mapping.value(value_image, inverse=True), value)
        value = {'o': 3}
        value_image = (3,)
        self.assertEqual(self.mapping.value(value), value_image)
        self.assertEqual(self.mapping.value(value_image, inverse=True), value)
        value = {'o': 0}
        value_image = {'o': 0}
        self.assertRaises(ValueError, self.mapping.value, value)
        self.assertRaises(ValueError,
                          lambda x: self.mapping.value(x, inverse=True),
                          value_image)


class TestOffsetIdentityMapping(TestIdentityMapping):
    def setUp(self):
        self.domain = Space({'o': Integer(2, 3)})
        self.mapping = OffsetIdentityMapping(self.domain)

    def test_value(self):
        value = {'o': 2}
        value_image = (0,)
        self.assertEqual(self.mapping.value(value), value_image)
        self.assertEqual(self.mapping.value(value_image, inverse=True), value)
        value = {'o': 3}
        value_image = (1,)
        self.assertEqual(self.mapping.value(value), value_image)
        self.assertEqual(self.mapping.value(value_image, inverse=True), value)
        value = {'o': 0}
        value_image = (3,)
        self.assertRaises(ValueError, self.mapping.value, value)
        self.assertRaises(ValueError,
                          lambda x: self.mapping.value(x, inverse=True),
                          value_image)


if __name__ == '__main__':
    unittest.main()
