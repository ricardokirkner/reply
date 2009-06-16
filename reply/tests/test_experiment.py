import unittest

from reply.experiment import Experiment


class GlueExperiment(Experiment):
    def init(self):
        return 'init'

    def start(self):
        return 'start'

    def step(self):
        roat = {'terminal': True}
        return roat

    def episode(self):
        return 'episode'

    def return_reward(self):
        return 'return_reward'

    def num_steps(self):
        return 'num_steps'

    def cleanup(self):
        return 'cleanup'

    def agent_message(self, message):
        return 'agent_message'

    def env_message(self, message):
        return 'env_message'


class TestExperiment(unittest.TestCase):

    def setUp(self):
        self.experiment = Experiment()
        self.glue_experiment = GlueExperiment()

    def test_set_glue_experiment(self):
        self.experiment.set_glue_experiment(self.glue_experiment)
        self.assertEqual(self.experiment.glue_experiment, self.glue_experiment)

    def test_init_not_initialized(self):
        self.assertRaises(NotImplementedError, self.experiment.init)

    def test_init(self):
        self.experiment.set_glue_experiment(self.glue_experiment)
        self.assertEqual(self.experiment.init(), 'init')

    def test_start_not_initialized(self):
        self.assertRaises(NotImplementedError, self.experiment.start)

    def test_start(self):
        self.experiment.set_glue_experiment(self.glue_experiment)
        self.assertEqual(self.experiment.start(), 'start')

    def test_step_not_initialized(self):
        self.assertRaises(NotImplementedError, self.experiment.step)

    def test_step(self):
        self.experiment.set_glue_experiment(self.glue_experiment)
        self.assertEqual(self.experiment.step(), {'terminal': True})

    def test_episode_not_initialized(self):
        self.assertRaises(NotImplementedError, self.experiment.episode)

    def test_episode(self):
        self.experiment.set_glue_experiment(self.glue_experiment)
        self.assertEqual(self.experiment.episode(), 'episode')

    def test_return_reward_not_initialized(self):
        self.assertRaises(NotImplementedError, self.experiment.return_reward)

    def test_return_reward(self):
        self.experiment.set_glue_experiment(self.glue_experiment)
        self.assertEqual(self.experiment.return_reward(), 'return_reward')

    def test_num_steps_not_initialized(self):
        self.assertRaises(NotImplementedError, self.experiment.num_steps)

    def test_num_steps(self):
        self.experiment.set_glue_experiment(self.glue_experiment)
        self.assertEqual(self.experiment.num_steps(), 'num_steps')

    def test_cleanup_not_initialized(self):
        self.assertRaises(NotImplementedError, self.experiment.cleanup)

    def test_cleanup(self):
        self.experiment.set_glue_experiment(self.glue_experiment)
        self.assertEqual(self.experiment.cleanup(), 'cleanup')

    def test_agent_message_not_initialized(self):
        self.assertRaises(NotImplementedError, self.experiment.agent_message, None)

    def test_agent_message(self):
        self.experiment.set_glue_experiment(self.glue_experiment)
        self.assertEqual(self.experiment.agent_message(None), 'agent_message')

    def test_env_message_not_initialized(self):
        self.assertRaises(NotImplementedError, self.experiment.env_message, None)

    def test_env_message(self):
        self.experiment.set_glue_experiment(self.glue_experiment)
        self.assertEqual(self.experiment.env_message(None), 'env_message')

    def test_run_not_initialized(self):
        self.assertRaises(NotImplementedError, self.experiment.run)

    def test_run(self):
        self.experiment.set_glue_experiment(self.glue_experiment)
        self.assertEqual(self.experiment.run(), None)


if __name__ == '__main__':
    unittest.main()

