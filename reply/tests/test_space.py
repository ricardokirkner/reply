import unittest

from reply.types import Char, Double, Integer, Space


class TestSpace(unittest.TestCase):

    def test_space_spec(self):
        spec = {'a': Integer(0, 3),
                'b': Double(0.0, 1.0),
                'c': Char()}
        space = Space(spec)
        self.assertEqual(space.spec, spec)

    def test_space_builder(self):
        spec = {'a': Integer(0, 3),
                'b': Double(0.0, 1.0),
                'c': Char()}
        space = Space(spec)
        self.assertEqual(space[Integer], {'a': spec['a']})
        self.assertEqual(space[Double], {'b': spec['b']})
        self.assertEqual(space[Char], {'c': spec['c']})


if __name__ == '__main__':
    unittest.main()
