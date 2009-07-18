import unittest

from reply import base

class TestBase(unittest.TestCase):
    def test_parameter_set(self):
        class T(base.AgentComponent):
            p = base.Parameter("docstring")

        class MyAgent(object):
            p = 2

        t = T(MyAgent())
        self.assertEqual(t.p, 2)

    def test_parameter_default(self):
        class T(base.AgentComponent):
            p = base.Parameter("docstring", 3)

        class MyAgent(object):
            pass

        t = T(MyAgent())
        self.assertEqual(t.p, 3)

    def test_parameter_override_default(self):
        class T(base.AgentComponent):
            p = base.Parameter("docstring", 3)

        class MyAgent(object):
            p = 4

        t = T(MyAgent())
        self.assertEqual(t.p, 4)

    def test_parameter_require_value(self):
        class T(base.AgentComponent):
            p = base.Parameter("docstring")

        class MyAgent(object):
            pass

        self.assertRaises(AttributeError, T, MyAgent())
