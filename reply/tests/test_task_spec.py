import unittest

from reply.types import Char, Double, Integer
from reply.util import TaskSpec

class TestTaskSpec(unittest.TestCase):

    def test_task_spec_parser(self):
        task_spec_str = "VERSION RL-Glue-3.0 PROBLEMTYPE episodic " \
            "DISCOUNTFACTOR 1 OBSERVATIONS INTS (3 0 1) " \
            "DOUBLES (2 -1.2 0.5) (-.07 .07) CHARCOUNT 0 " \
            "ACTIONS INTS (0 4)  REWARDS (-5.0 5.0) " \
            "EXTRA some other stuff goes here"
        task_spec = TaskSpec.parse(task_spec_str)
        self.assertEqual(task_spec.version, 'RL-Glue-3.0')
        self.assertEqual(task_spec.problem_type, 'episodic')
        self.assertEqual(task_spec.discount_factor, 1)

        integers = [Integer(0, 1)] * 3
        doubles = [Double(-1.2, 0.5)] * 2 + [Double(-0.07, 0.07)]
        chars = [Char()]*0
        self.assertEqual(task_spec.observations, {Integer: integers,
                                                  Double: doubles,
                                                  Char: chars})

        integers = [Integer(0, 4)]
        self.assertEqual(task_spec.actions, {Integer: integers,
                                             Double: [],
                                             Char: []})

        self.assertEqual(task_spec.rewards, Double(-5.0, 5.0))
        self.assertEqual(task_spec.extra, "some other stuff goes here")

    def test_task_spec_builder(self):
        version = 'RL-Glue-3.0'
        problem_type = 'episodic'
        discount_factor = 1
        observations = {Integer: [Integer(0, 1)] * 3,
                        Double: [Double(-1.2, 0.5)] * 2 + [Double(-0.07, 0.07)],
                        Char: [Char()] * 0}
        actions = {Integer: [Integer(0, 4)], Double: [], Char: []}
        rewards = Double(-5.0, 5.0)
        extra = "some other stuff goes here"
        task_spec = TaskSpec(version, problem_type, discount_factor,
                             observations, actions, rewards, extra)

        self.assertEqual(task_spec.version, version)
        self.assertEqual(task_spec.problem_type, problem_type)
        self.assertEqual(task_spec.discount_factor, discount_factor)
        self.assertEqual(task_spec.observations, observations)
        self.assertEqual(task_spec.actions, actions)
        self.assertEqual(task_spec.rewards, rewards)
        self.assertEqual(task_spec.extra, extra)

    def test_task_spec_string(self):
        task_spec_str = "VERSION RL-Glue-3.0 PROBLEMTYPE episodic " \
            "DISCOUNTFACTOR 1 OBSERVATIONS INTS (0 1) (0 1) (0 1) " \
            "DOUBLES (-1.2 0.5) (-1.2 0.5) (-0.07 0.07) CHARCOUNT 2 " \
            "ACTIONS INTS (0 4) REWARDS (-5.0 5.0) " \
            "EXTRA some other stuff goes here"
        version = 'RL-Glue-3.0'
        problem_type = 'episodic'
        discount_factor = 1
        observations = {Integer: [Integer(0, 1)] * 3,
                        Double: [Double(-1.2, 0.5)] * 2 + [Double(-0.07, 0.07)],
                        Char: [Char()] * 2}
        actions = {Integer: [Integer(0, 4)], Double: [], Char: []}
        rewards = Double(-5.0, 5.0)
        extra = "some other stuff goes here"
        task_spec = TaskSpec(version, problem_type, discount_factor,
                             observations, actions, rewards, extra)

        self.assertEqual(str(task_spec), task_spec_str)

    def test_task_spec_order(self):
        assert False



if __name__ == '__main__':
    unittest.main()
