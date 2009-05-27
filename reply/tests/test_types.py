import random
import unittest

from reply.types import Char, Double, Integer, Number, Space


class TestNumber(unittest.TestCase):
    def test_number(self):
        min, max = random.randint(-20, 20), random.randint(-20, 20)
        n = Number(min, max)
        self.assertEqual(n.min, min)
        self.assertEqual(n.max, max)

    def test_number_equal(self):
        min, max = random.randint(-20, 20), random.randint(-20, 20)
        n1 = Number(min, max)
        n2 = Number(n1.min, n1.max)
        self.assertEqual(n1, n2)


class TestInteger(unittest.TestCase):
    def test_integer_str(self):
        min, max = random.randint(-20, 20), random.randint(-20, 20)
        n = Integer(min, max)
        self.assertEqual(str(n), "(%i %i)" % (n.min, n.max))


class TestDouble(unittest.TestCase):
    def test_double(self):
        min, max = random.random() * 20, random.random() * 20
        n = Double(min, max)
        self.assertEqual(n.min, min)
        self.assertEqual(n.max, max)

    def test_double_equal(self):
        min, max = random.random() * 20, random.random() * 20
        n1 = Double(min, max)
        n2 = Double(n1.min, n1.max)
        self.assertEqual(n1, n2)

    def test_double_str(self):
        min, max = random.random() * 20, random.random() * 20
        n = Double(min, max)
        self.assertEqual(str(n), "(%f %f)" % (n.min, n.max))


class TestChar(unittest.TestCase):
    def test_char_equal(self):
        c1 = Char()
        c2 = Char()
        self.assertEqual(c1, c2)


class TestSpace(unittest.TestCase):
    def test_space_spec(self):
        spec = dict(a=5)
        s = Space(spec)
        self.assertEqual(s.spec, spec)

    def test_space_getitem(self):
        spec = dict(value=Integer(0, 10))
        s = Space(spec)
        self.assertEqual(s[Integer], spec)

    def test_space_getitem_mixed(self):
        spec = {'a': Integer(0, 10),
                'b': Integer(-20, -3),
                'c': Integer(-5, 5),
                'd': Double(-0.05, 20.3),
                'e': Double(12, 13),
                'f': Double(0, 0.125),
                'g': Char(),
                'h': Char(),
                'i': Char()}
        s = Space(spec)
        self.assertEqual(s[Integer], {'a': Integer(0, 10),
                                      'b': Integer(-20, -3),
                                      'c': Integer(-5, 5)})
        self.assertEqual(s[Double], {'d': Double(-0.05, 20.3),
                                     'e': Double(12, 13),
                                     'f': Double(0, 0.125)})
        self.assertEqual(s[Char], {'g': Char(),
                                   'h': Char(),
                                   'i': Char()})

    def test_space_str_one_int(self):
        s = Space(dict(value=Integer(0, 10)))
        self.assertEqual(str(s), "INTS (0 10)")

    def test_space_str_one_int2(self):
        s = Space(dict(value=Integer(-20, 10)))
        self.assertEqual(str(s), "INTS (-20 10)")

    def test_space_str_two_int(self):
        s = Space(dict(varname_a=Integer(0, 10), varname_b=Integer(0, 1)))
        self.assertEqual(str(s), "INTS (0 10) (0 1)")

    def test_space_str_two_int_two(self):
        s = Space(dict(varname_b=Integer(0, 10), varname_a=Integer(0, 1)))
        self.assertEqual(str(s), "INTS (0 1) (0 10)")

    def test_space_str_one_double(self):
        s = Space(dict(value=Double(0, 10)))
        self.assertEqual(str(s), "DOUBLES (0.000000 10.000000)")

    def test_space_str_one_double2(self):
        s = Space(dict(value=Double(-20, 10)))
        self.assertEqual(str(s), "DOUBLES (-20.000000 10.000000)")

    def test_space_str_two_double(self):
        s = Space(dict(varname_a=Double(0, 10), varname_b=Double(0, 1)))
        self.assertEqual(str(s),
                         "DOUBLES (0.000000 10.000000) (0.000000 1.000000)")

    def test_space_str_two_double_two(self):
        s = Space(dict(varname_b=Double(0, 10), varname_a=Double(0, 1)))
        self.assertEqual(str(s),
                         "DOUBLES (0.000000 1.000000) (0.000000 10.000000)")

    def test_space_str_one_char(self):
        s = Space(dict(varname_a=Char()))
        self.assertEqual(str(s), "CHARCOUNT 1")

    def test_space_str_two_char(self):
        s = Space(dict(varname_a=Char(), varname_b=Char()))
        self.assertEqual(str(s), "CHARCOUNT 2")

    def test_space_str_char_int(self):
        s = Space(dict(varname_a=Char(), varname_b=Integer(1, 10)))
        self.assertEqual(str(s), "INTS (1 10) CHARCOUNT 1")

    def test_space_str_mixed(self):
        spec = {'a': Integer(0, 10),
                'b': Integer(-20, -3),
                'c': Integer(-5, 5),
                'd': Double(-0.05, 20.3),
                'e': Double(12, 13),
                'f': Double(0, 0.125),
                'g': Char(),
                'h': Char(),
                'i': Char()}
        s = Space(spec)
        self.assertEqual(str(s), "INTS (0 10) (-20 -3) (-5 5) " \
            "DOUBLES (-0.050000 20.300000) (12.000000 13.000000) " \
            "(0.000000 0.125000) CHARCOUNT 3")


if __name__ == '__main__':
    unittest.main()
