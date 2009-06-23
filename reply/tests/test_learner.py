import unittest

from reply.learner import Learner, QLearner


class TestLearner(unittest.TestCase):
    def setUp(self):
        self.policy = None
        self.learner = Learner(self.policy)

    def test_learner_builder(self):
        self.assertEqual(self.learner.policy, self.policy)

    def test_learner_update(self):
        self.assertRaises(NotImplementedError, self.learner.update,
                          None, None, None, None)


class TestQLearner(TestLearner):
    def test_learner_builder(self):
        policy = None
        learning_rate = 1.0
        learning_rate_decay = 0.9
        learning_rate_min = 0.1
        value_discount = 0.9
        learner = QLearner(policy, learning_rate, learning_rate_decay,
                           learning_rate_min, value_discount)
        self.assertEqual(learner.policy, policy)
        self.assertEqual(learner.learning_rate, learning_rate)
        self.assertEqual(learner.learning_rate_decay, learning_rate_decay)
        self.assertEqual(learner.learning_rate_min, learning_rate_min)
        self.assertEqual(learner.value_discount, value_discount)

    def test_learner_update(self):
        pass


#class TestSarsaLearner(unittest.TestCase):
#    def test_learner_builder(self):
#        pass
#
#    def test_learner_update(self):
#        pass


if __name__ == '__main__':
    unittest.main()
