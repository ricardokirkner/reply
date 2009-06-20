import random
import unittest

from reply.datatypes import Char, Double, Integer, Model, Number, Space


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


class TestModel(unittest.TestCase):
    def test_model_builder(self):
        observations_spec = {'a': Integer(0, 1)}
        actions_spec = {'b': Integer(0, 1)}
        model = Model({'spec': observations_spec}, {'spec': actions_spec})
        self.assertEqual(model.observations, Space(observations_spec))
        self.assertEqual(model.actions, Space(actions_spec))

    def test_model_build_space(self):
        spec = {'a': Integer(0, 1)}
        space = Space(spec)
        model = Model()
        result = model._build_space({'spec': spec})
        self.assertEqual(result, Space(spec))
        result = model._build_space(space)
        self.assertEqual(result, space)

    def test_model_equal(self):
        spec = {'a': Integer(0, 1)}
        model1 = Model({'spec': spec})
        model2 = Model({'spec': spec})
        model3 = Model(None, {'spec': spec})
        model4 = Model()
        self.assertEqual(model1, model2)
        self.assertNotEqual(model1, model3)
        self.assertNotEqual(model1, model4)
        self.assertNotEqual(model3, model4)


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

    def test_space_getitem_key(self):
        spec = {'value': Integer(0, 1)}
        s = Space(spec)
        self.assertTrue(Integer in s)

    def test_space_getitem_value(self):
        spec = {'value': Integer(0, 1)}
        s = Space(spec)
        self.assertTrue('value' in s)

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

    def test_space_str_order(self):
        spec = {'a': Integer(0, 10),
                'b': Integer(2, 4),
                'c': Double(0.0, 1.0),
                'd': Double(1.0, 2.0),
                'e': Char(),
                'f': Char()}
        order = {Integer: ['b', 'a'],
                 Double: ['c', 'd']}
        s = Space(spec, order)
        self.assertEqual(str(s), "INTS (2 4) (0 10) " \
            "DOUBLES (0.000000 1.000000) (1.000000 2.000000) CHARCOUNT 2")

    def test_space_equal_different_keys(self):
        spec1 = {'a': Integer(0, 10)}
        spec2 = {'b': Integer(0, 10)}
        space1 = Space(spec1)
        space2 = Space(spec2)
        self.assertNotEqual(space1, space2)

    def test_space_equal_spec_same(self):
        spec = {'a': Integer(0, 10)}
        space1 = Space(spec)
        space2 = Space(spec)
        self.assertEqual(space1, space2)

    def test_space_equal_spec_equal(self):
        spec1 = {'a': Integer(0, 10)}
        spec2 = {'a': Integer(0, 10)}
        space1 = Space(spec1)
        space2 = Space(spec2)
        self.assertEqual(space1, space2)

    def test_space_equal_spec_equivalent(self):
        spec1 = {'a': Integer(0, 10)}
        spec2 = {'a': Integer(0.0, 10.0)}
        space1 = Space(spec1)
        space2 = Space(spec2)
        self.assertEqual(space1, space2)

    def test_space_equal_spec_different(self):
        spec1 = {'a': Integer(0, 10)}
        spec2 = {'a': Integer(0, 1)}
        space1 = Space(spec1)
        space2 = Space(spec2)
        self.assertNotEqual(space1, space2)

    def test_space_equal_spec_different_type(self):
        spec1 = {'a': Integer(0, 10)}
        spec2 = {'a': Double(0, 10)}
        space1 = Space(spec1)
        space2 = Space(spec2)
        self.assertNotEqual(space1, space2)

    def test_space_size(self):
        spec1 = {'a': Integer(0, 1)}
        spec2 = {'a': Integer(0, 1), '': [Integer(1,2)]}
        space1 = Space(spec1)
        space2 = Space(spec2)
        self.assertEqual(space1.size, 2)
        self.assertEqual(space2.size, 4)

    def test_space_get_names_spec(self):
        spec = {'a': Integer(0, 1),
                'b': Double(0.0, 1.0),
                'c': Char()}
        space = Space(spec)
        self.assertEqual(space.get_names_spec(), "INTS a DOUBLES b CHARS c")

    def test_space_get_names_list(self):
        spec = {'a': Integer(0, 1),
                'b': Integer(1, 2),
                'c': Double(0.0, 1.0),
                'd': Double(1.0, 2.0),
                'e': Char(),
                'f': Char()}
        space = Space(spec)
        ints = space.get_names_list(Integer)
        doubles = space.get_names_list(Double)
        chars = space.get_names_list(Char)
        names = space.get_names_list()
        self.assertEqual(ints, ['a', 'b'])
        self.assertEqual(doubles, ['c', 'd'])
        self.assertEqual(chars, ['e', 'f'])
        self.assertEqual(names, ['a', 'b', 'c', 'd', 'e', 'f'])

    def test_space_get_names_list_order(self):
        spec = {'a': Integer(0, 1),
                'b': Integer(1, 2),
                'c': Double(0.0, 1.0),
                'd': Double(1.0, 2.0),
                'e': Char(),
                'f': Char()}
        order = {Integer: ['b', 'a'],
                 Double: ['d', 'c'],
                 Char: ['f', 'e']}
        space = Space(spec, order)
        ints = space.get_names_list(Integer)
        doubles = space.get_names_list(Double)
        chars = space.get_names_list(Char)
        names = space.get_names_list()
        self.assertEqual(ints, ['b', 'a'])
        self.assertEqual(doubles, ['d', 'c'])
        self.assertEqual(chars, ['f', 'e'])
        self.assertEqual(names, ['b', 'a', 'd', 'c', 'f', 'e'])




if __name__ == '__main__':
    unittest.main()
