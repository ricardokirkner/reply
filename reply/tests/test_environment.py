import unittest

from reply import environment
from reply.types import *

class TestEnviron(unittest.TestCase):
    def test_env(self):
        class MyEnv(environment.Environment):
            action_space = dict(choice=Integer(0,10))
            observation_space = dict(state=Integer(0,0))
            problem_type = "episodic"
            discount_factor = 1.0
            rewards = -1, 1

        e = MyEnv()
        self.assertEqual(e.get_task_spec(), "VERSION RL-Glue-3.0 PROBLEMTYPE"
            " episodic DISCOUNTFACTOR 1.0 OBSERVATIONS INTS (0 0) ACTIONS "
            "INTS (0 10) REWARDS (-1 1) EXTRA None")


if __name__ == '__main__':
    unittest.main()
