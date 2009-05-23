import unittest

from rlglue.types import Action, Observation, Reward_observation_terminal

from reply.glue import adapt
from reply.types import Char, Double, Integer, Space


class TestRLGlue(unittest.TestCase):

    def test_adapt_to_action(self):
        space = Space(dict(choice=Integer(0, 1)))
        reply_action = dict(choice=0)
        action = adapt(reply_action, space, Action)

        expected_action = Action()
        expected_action.intArray = [0]

        self.assertTrue(action.sameAs(expected_action))

    def test_adapt_from_action(self):
        space = Space(dict(choice=Integer(0, 1)))
        action = Action()
        action.intArray = [0]
        reply_action = adapt(action, space)

        expected_action = dict(choice=0)

        self.assertEquals(reply_action, expected_action)


if __name__ == '__main__':
    unittest.main()
