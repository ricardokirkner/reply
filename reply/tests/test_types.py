import unittest

from reply import types


class TestSpace(unittest.TestCase):
    def test_one_int(self):
        s = types.Space(dict(value=types.Integer(0,10)))
        self.assertEqual(str(s), "INTS (0 10)")

    def test_one_int2(self):
        s = types.Space(dict(value=types.Integer(-20,10)))
        self.assertEqual(str(s), "INTS (-20 10)")

    def test_two_int(self):
        s = types.Space(dict(
            varname_a=types.Integer(0,10),
            varname_b=types.Integer(0,1)))
        self.assertEqual(str(s),
                "INTS (0 10) (0 1)")

    def test_two_int_two(self):
        s = types.Space(dict(
            varname_b=types.Integer(0,10),
            varname_a=types.Integer(0,1)))
        self.assertEqual(str(s),
                "INTS (0 1) (0 10)")

    def test_one_double(self):
        s = types.Space(dict(value=types.Double(0,10)))
        self.assertEqual(str(s), "DOUBLES (0.000000 10.000000)")

    def test_one_double2(self):
        s = types.Space(dict(value=types.Double(-20,10)))
        self.assertEqual(str(s), "DOUBLES (-20.000000 10.000000)")

    def test_two_double(self):
        s = types.Space(dict(
            varname_a=types.Double(0,10),
            varname_b=types.Double(0,1)))
        self.assertEqual(str(s),
                "DOUBLES (0.000000 10.000000) (0.000000 1.000000)")

    def test_two_double_two(self):
        s = types.Space(dict(
            varname_b=types.Double(0,10),
            varname_a=types.Double(0,1)))
        self.assertEqual(str(s),
                "DOUBLES (0.000000 1.000000) (0.000000 10.000000)")

    def test_one_char(self):
        s = types.Space(dict(varname_a=types.Char()))
        self.assertEqual(str(s),
                "CHARCOUNT 1")

    def test_two_char(self):
        s = types.Space(dict(varname_a=types.Char(), varname_b=types.Char()))
        self.assertEqual(str(s),
                "CHARCOUNT 2")

    def test_char_int(self):
        s = types.Space(dict(varname_a=types.Char(),
                             varname_b=types.Integer(1,10)))
        self.assertEqual(str(s),
                "INTS (1 10) CHARCOUNT 1")


if __name__ == '__main__':
    unittest.main()
