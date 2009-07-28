import unittest

from reply.datatypes import Integer, Space, Double
from reply.mapping import IdentityMapping, Mapping, OffsetIdentityMapping, \
    TileMapping


class TestMapping(unittest.TestCase):
    def setUp(self):
        self.domain = Space({'o': Integer(2, 3)})
        self.image = Space({'o': Integer(0, 1)})
        self.mapping = Mapping(self.domain, self.image)

    def test_builder(self):
        self.assertEqual(self.mapping.domain, self.domain)
        self.assertEqual(self.mapping.image, self.image)

    def test_value(self):
        self.assertRaises(NotImplementedError, self.mapping.value, {'o':2})
        self.assertRaises(NotImplementedError,
                          lambda x: self.mapping.value(x, inverse=True), {'o':1})


class TestIdentityMapping(unittest.TestCase):
    def setUp(self):
        self.domain = Space({'o': Integer(2, 3)})
        self.mapping = IdentityMapping(self.domain)

    def test_builder(self):
        self.assertEqual(self.mapping.domain, self.domain)
        self.assertEqual(self.mapping.image, self.domain)

    def test_value(self):
        value = {'o': 2}
        value_image = {'o': 2}
        self.assertEqual(self.mapping.value(value), value_image)
        self.assertEqual(self.mapping.value(value_image, inverse=True), value)
        value = {'o': 3}
        value_image = {'o': 3}
        self.assertEqual(self.mapping.value(value), value_image)
        self.assertEqual(self.mapping.value(value_image, inverse=True), value)
        value = {'o': 0}
        value_image = {'o': 0}
        self.assertRaises(ValueError, self.mapping.value, value)
        self.assertRaises(ValueError,
                          lambda x: self.mapping.value(x, inverse=True),
                          value_image)


class TestOffsetIdentityMapping(unittest.TestCase):
    def setUp(self):
        self.domain = Space({'o': Integer(2, 3)})
        self.mapping = OffsetIdentityMapping(self.domain)

    def test_value(self):
        value = {'o': 2}
        value_image = {'o': 0}
        self.assertEqual(self.mapping.value(value), value_image)
        self.assertEqual(self.mapping.value(value_image, inverse=True), value)
        value = {'o': 3}
        value_image = {'o': 1}
        self.assertEqual(self.mapping.value(value), value_image)
        self.assertEqual(self.mapping.value(value_image, inverse=True), value)
        value = {'o': 0}
        value_image = {'o': 3}
        self.assertRaises(ValueError, self.mapping.value, value)
        self.assertRaises(ValueError,
                          lambda x: self.mapping.value(x, inverse=True),
                          value_image)


class TestTileMapping(unittest.TestCase):
    def setUp(self):
        self.domain = Space({'o': Double(2, 3)})
        self.mapping = TileMapping(self.domain, {'o': 3})

    def test_value(self):

        value = {'o': 2}
        value_image = {'o': 0}
        self.assertEqual(self.mapping.value(value), value_image)
        self.assert_(self.mapping.value(value_image, inverse=True)['o'] - value['o'] < 1./3)
        value = {'o': 3}
        value_image = {'o': 2}
        self.assertEqual(self.mapping.value(value), value_image)
        self.assert_(self.mapping.value(value_image, inverse=True)['o'] > 2.5)
        value = {'o': 0}
        value_image = {'o': 4}
        self.assertRaises(ValueError, self.mapping.value, value)
        self.assertRaises(ValueError,
                          lambda x: self.mapping.value(x, inverse=True),
                          value_image)

class TestTileMapping2(unittest.TestCase):
    def setUp(self):
        self.domain = Space({'o': Double(-2, 2)})
        self.mapping = TileMapping(self.domain, {'o': 100})

    def test_inverse(self):
        for i in range(int(self.mapping.image.o.max)):
            o = {'o': i}
            self.assertEqual(o,
                             self.mapping.value(self.mapping._inverse(o)))

    def test_inverse(self):
        for i in range(int(self.mapping.image.o.max)):
            o = {'o': i}
            print o, self.mapping.value(self.mapping._inverse(o))
            self.assertEqual(o,
                             self.mapping.value(self.mapping._inverse(o)))

if __name__ == '__main__':
    unittest.main()
