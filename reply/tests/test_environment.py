import unittest

from reply.environment import Environment
from reply.types import Char, Double, Integer, Space
from reply.util import TaskSpec

class TestEnviron(unittest.TestCase):

    def setUp(self):
        self.environment = Environment()

    def test_environment_builder(self):
        self.assertEqual(self.environment.initialized, False)
        self.assertEqual(self.environment.problem_type, 'episodic')
        self.assertEqual(self.environment.discount_factor, 1.0)
        self.assertEqual(self.environment.rewards, (0, 1))
        self.assertEqual(self.environment.actions_spec, {})
        self.assertEqual(self.environment.observations_spec, {})

    def test_environment_set_action_space_already_initialized(self):
        self.environment.init()
        spec = {'a1': Integer(0, 1)}
        self.assertRaises(Exception, self.environment.set_action_space, **spec)

    def test_environment_set_observation_space_already_initialized(self):
        self.environment.init()
        spec = {'o1': Integer(0, 1)}
        self.assertRaises(Exception, self.environment.set_observation_space,
                          **spec)

    def test_environment_get_task_spec(self):
        problem_type = self.environment.problem_type
        discount_factor = self.environment.discount_factor
        action_space = self.environment._action_space
        observation_space = self.environment._observation_space
        rewards = self.environment.rewards
        extra = self.environment.__doc__
        task_spec = TaskSpec(problem_type=problem_type,
                             discount_factor=discount_factor,
                             observations=observation_space,
                             actions=action_space,
                             rewards=rewards,
                             extra=extra)
        self.assertEqual(self.environment.get_task_spec(), task_spec)

    def test_environment_init(self):
        problem_type = self.environment.problem_type
        discount_factor = self.environment.discount_factor
        action_space = self.environment._action_space
        observation_space = self.environment._observation_space
        rewards = self.environment.rewards
        extra = self.environment.__doc__
        default_task_spec = TaskSpec(problem_type=problem_type,
                                     discount_factor=discount_factor,
                                     observations=observation_space,
                                     actions=action_space,
                                     rewards=rewards,
                                     extra=extra)
        task_spec = self.environment.init()
        self.assertEqual(self.environment.initialized, True)
        self.assertEqual(task_spec, default_task_spec)

    def test_environment_start(self):
        def start():
            return {'o1': 1}
        self.environment._start = start
        observation = self.environment.start()
        self.assertEqual(self.environment.started, True)
        self.assertEqual(observation, {'o1': 1})

    def test_environment_step(self):
        def step(action):
            return {'o1': 1}
        self.environment._step = step
        observation = self.environment.step({})
        self.assertEqual(observation, {'o1': 1})

    def test_environment_cleanup(self):
        self.assertEqual(self.environment.cleanup(), None)


if __name__ == '__main__':
    unittest.main()
