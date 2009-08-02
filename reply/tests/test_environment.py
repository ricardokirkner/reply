import unittest

from reply.environment import Environment
from reply.datatypes import Char, Double, Integer, Model, Space
from reply.util import TaskSpec

class TestEnviron(unittest.TestCase):

    def setUp(self):
        self.environment = Environment()

    def test_environment_builder(self):
        self.assertEqual(self.environment.initialized, False)
        self.assertEqual(self.environment.problem_type, 'episodic')
        self.assertEqual(self.environment.discount_factor, 1.0)
        self.assertEqual(self.environment.rewards, Double(0, 1))
        self.assertEqual(self.environment.model, Model())

    def test_environment_get_task_spec(self):
        problem_type = self.environment.problem_type
        discount_factor = self.environment.discount_factor
        action_space = self.environment.model.actions
        observation_space = self.environment.model.observations
        rewards = self.environment.rewards
        extra = ''
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
        action_space = self.environment.model.actions
        observation_space = self.environment.model.observations
        rewards = self.environment.rewards
        extra = ''
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
        observation = self.environment.start()
        self.assertEqual(observation, {})

    def test_environment_step(self):
        observation = self.environment.step({})
        self.assertEqual(observation, {'reward': 0, 'terminal': False})

    def test_environment_cleanup(self):
        self.assertEqual(self.environment.cleanup(), None)


if __name__ == '__main__':
    unittest.main()
