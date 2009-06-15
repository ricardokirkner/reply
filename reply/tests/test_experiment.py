import unittest

from reply.experiment import Experiment


class TestExperiment(unittest.TestCase):

    def setUp(self):
        self.experiment = Experiment()

    def test_init(self):
        self.experiment.init()
        self.assertEqual(self.experiment.initialized, True)

    def test_start(self):
        self.experiment.start()
        self.assertEqual(self.experiment.started, True)

    def test_step(self):
        self.assertEqual(self.experiment.step(), None)

    def test_run(self):
        self.assertEqual(self.experiment.run(), None)

    def test_cleanup(self):
        self.assertEqual(self.experiment.cleanup(), None)
        self.assertEqual(self.experiment.started, False)
        self.assertEqual(self.experiment.initialized, False)


if __name__ == '__main__':
    unittest.main()

