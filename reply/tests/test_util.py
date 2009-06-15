import unittest
import simplejson

from rlglue.utils.TaskSpecVRLGLUE3 import TaskSpecParser

from reply.datatypes import Char, Double, Integer, Space
from reply.util import parse_spaces, build_spec, parse_names
from reply.util import TaskSpec, MessageHandler


class TestUtil(unittest.TestCase):
    def test_parse_spaces(self):
        task_spec_str = "VERSION RL-Glue-3.0 PROBLEMTYPE episodic " \
            "DISCOUNTFACTOR 1 OBSERVATIONS INTS (3 0 1) " \
            "DOUBLES (2 -1.2 0.5) (-.07 .07) CHARCOUNT 2 " \
            "ACTIONS INTS (0 4)  REWARDS (-5.0 5.0) " \
            "EXTRA some other stuff goes here " \
            "OBSERVATIONS INTS oi1 oi2 oi3 DOUBLES od1 od2 od3 " \
            "CHARS oc1 oc2 ACTIONS INTS ai1"
        data = "some other stuff goes here OBSERVATIONS INTS oi1 oi2 oi3 " \
            "DOUBLES od1 od2 od3 CHARS oc1 oc2 ACTIONS INTS ai1"
        parser = TaskSpecParser(task_spec_str)
        observations, actions = parse_spaces(data, parser)

        expected_observations = Space({'oi1': Integer(0, 1),
                                       'oi2': Integer(0, 1),
                                       'oi3': Integer(0, 1),
                                       'od1': Double(-1.2, 0.5),
                                       'od2': Double(-1.2, 0.5),
                                       'od3': Double(-0.07, 0.07),
                                       'oc1': Char(),
                                       'oc2': Char()})
        expected_actions = Space({'ai1': Integer(0, 4)})

        self.assertEqual(observations, expected_observations)
        self.assertEqual(actions, expected_actions)

    def test_parse_spaces_no_actions(self):
        task_spec_str = "VERSION RL-Glue-3.0 PROBLEMTYPE episodic " \
            "DISCOUNTFACTOR 1 OBSERVATIONS INTS (3 0 1) " \
            "DOUBLES (2 -1.2 0.5) (-.07 .07) CHARCOUNT 2 " \
            "ACTIONS INTS (0 4)  REWARDS (-5.0 5.0) " \
            "EXTRA some other stuff goes here " \
            "OBSERVATIONS INTS oi1 oi2 oi3 DOUBLES od1 od2 od3 " \
            "CHARS oc1 oc2"
        data = "some other stuff goes here OBSERVATIONS INTS oi1 oi2 oi3 " \
            "DOUBLES od1 od2 od3 CHARS oc1 oc2"
        parser = TaskSpecParser(task_spec_str)
        observations, actions = parse_spaces(data, parser)

        expected_observations = Space({'oi1': Integer(0, 1),
                                       'oi2': Integer(0, 1),
                                       'oi3': Integer(0, 1),
                                       'od1': Double(-1.2, 0.5),
                                       'od2': Double(-1.2, 0.5),
                                       'od3': Double(-0.07, 0.07),
                                       'oc1': Char(),
                                       'oc2': Char()})
        expected_actions = Space()

        self.assertEqual(observations, expected_observations)
        self.assertEqual(actions, expected_actions)

    def test_build_spec(self):
        names = ['INTS', 'a', 'DOUBLES', 'b', 'CHARS', 'c']
        values = {'INTS': [[0, 1]], 'DOUBLES': [[0.0, 1.0]], 'CHARS': 1}
        spec = build_spec(names, values)

        expected_spec = {'a': Integer(0, 1),
                         'b': Double(0.0, 1.0),
                         'c': Char()}

        self.assertEqual(spec, expected_spec)

    def test_parse_names(self):
        names = ['INTS', 'a', 'DOUBLES', 'b', 'CHARS', 'c']
        values = {'INTS': [[0, 1]], 'DOUBLES': [[0.0, 1.0]], 'CHARS': 1}
        result = {}
        parse_names(names, values, result)

        expected_result = {Integer: {'a': Integer(0, 1)},
                           Double: {'b': Double(0.0, 1.0)},
                           Char: {'c': Char()}}

        self.assertEqual(result, expected_result)


class TestTaskSpec(unittest.TestCase):

    def test_task_spec_parser_no_names(self):
        task_spec_str = "VERSION RL-Glue-3.0 PROBLEMTYPE episodic " \
            "DISCOUNTFACTOR 1 OBSERVATIONS INTS (3 0 1) " \
            "DOUBLES (2 -1.2 0.5) (-.07 .07) CHARCOUNT 0 " \
            "ACTIONS INTS (0 4)  REWARDS (-5.0 5.0) " \
            "EXTRA some other stuff goes here"
        task_spec = TaskSpec.parse(task_spec_str)
        self.assertEqual(task_spec.version, 'RL-Glue-3.0')
        self.assertEqual(task_spec.problem_type, 'episodic')
        self.assertEqual(task_spec.discount_factor, 1)

        observations = Space()
        actions = Space()
        self.assertEqual(task_spec.observations, observations)
        self.assertEqual(task_spec.actions, actions)

        self.assertEqual(task_spec.rewards, Double(-5.0, 5.0))
        self.assertEqual(task_spec.extra, "some other stuff goes here")

    def test_task_spec_parser_names(self):
        task_spec_str = "VERSION RL-Glue-3.0 PROBLEMTYPE episodic " \
            "DISCOUNTFACTOR 1 OBSERVATIONS INTS (3 0 1) " \
            "DOUBLES (2 -1.2 0.5) (-.07 .07) CHARCOUNT 2 " \
            "ACTIONS INTS (0 4)  REWARDS (-5.0 5.0) " \
            "EXTRA some other stuff goes here " \
            "OBSERVATIONS INTS oi1 oi2 oi3 DOUBLES od1 od2 od3 " \
            "CHARS oc1 oc2 ACTIONS INTS ai1"
        task_spec = TaskSpec.parse(task_spec_str)
        self.assertEqual(task_spec.version, 'RL-Glue-3.0')
        self.assertEqual(task_spec.problem_type, 'episodic')
        self.assertEqual(task_spec.discount_factor, 1)

        observations = Space({'oi1': Integer(0, 1), 'oi2': Integer(0, 1),
                              'oi3': Integer(0, 1),
                              'od1': Double(-1.2, 0.5),
                              'od2': Double(-1.2, 0.5),
                              'od3': Double(-0.07, 0.07),
                              'oc1': Char(), 'oc2': Char()})
        actions = Space({'ai1': Integer(0, 4)})
        self.assertEqual(task_spec.observations, observations)
        self.assertEqual(task_spec.actions, actions)

        self.assertEqual(task_spec.rewards, Double(-5.0, 5.0))
        extra = "some other stuff goes here OBSERVATIONS INTS oi1 oi2 oi3 " \
            "DOUBLES od1 od2 od3 CHARS oc1 oc2 ACTIONS INTS ai1"
        self.assertEqual(task_spec.extra, extra)

    def test_task_spec_parser_names_no_actions(self):
        task_spec_str = "VERSION RL-Glue-3.0 PROBLEMTYPE episodic " \
            "DISCOUNTFACTOR 1 OBSERVATIONS INTS (3 0 1) " \
            "DOUBLES (2 -1.2 0.5) (-.07 .07) CHARCOUNT 2 " \
            "ACTIONS INTS (0 4)  REWARDS (-5.0 5.0) " \
            "EXTRA some other stuff goes here " \
            "OBSERVATIONS INTS oi1 oi2 oi3 DOUBLES od1 od2 od3 " \
            "CHARS oc1 oc2"
        task_spec = TaskSpec.parse(task_spec_str)
        self.assertEqual(task_spec.version, 'RL-Glue-3.0')
        self.assertEqual(task_spec.problem_type, 'episodic')
        self.assertEqual(task_spec.discount_factor, 1)

        observations = Space({'oi1': Integer(0, 1), 'oi2': Integer(0, 1),
                              'oi3': Integer(0, 1),
                              'od1': Double(-1.2, 0.5),
                              'od2': Double(-1.2, 0.5),
                              'od3': Double(-0.07, 0.07),
                              'oc1': Char(), 'oc2': Char()})
        actions = Space()
        self.assertEqual(task_spec.observations, observations)
        self.assertEqual(task_spec.actions, actions)

        self.assertEqual(task_spec.rewards, Double(-5.0, 5.0))
        extra = "some other stuff goes here OBSERVATIONS INTS oi1 oi2 oi3 " \
            "DOUBLES od1 od2 od3 CHARS oc1 oc2"
        self.assertEqual(task_spec.extra, extra)

    def test_task_spec_builder(self):
        version = 'RL-Glue-3.0'
        problem_type = 'episodic'
        discount_factor = 1
        # FIXME: the API has changed. need to update this test to make sense
        #        need to generate unique names when no names have been defined
        #        in the EXTRA section
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

    def test_task_spec_string_no_names(self):
        task_spec_str = "VERSION RL-Glue-3.0 PROBLEMTYPE episodic " \
            "DISCOUNTFACTOR 1 OBSERVATIONS INTS (0 1) (0 1) (0 1) " \
            "DOUBLES (-1.200000 0.500000) (-1.200000 0.500000) (-0.070000 0.070000) CHARCOUNT 2 " \
            "ACTIONS INTS (0 4) REWARDS (-5.0 5.0) " \
            "EXTRA some other stuff goes here"
        version = 'RL-Glue-3.0'
        problem_type = 'episodic'
        discount_factor = 1
        observations = Space({'oi1': Integer(0, 1), 'oid2': Integer(0, 1),
                              'oi3': Integer(0, 1),
                              'od1': Double(-1.2, 0.5),
                              'od2': Double(-1.2, 0.5),
                              'od3': Double(-0.07, 0.07),
                              'oc1': Char(), 'oc2': Char()})
        actions = Space({'ai1': Integer(0, 4)})
        rewards = Double(-5.0, 5.0)
        extra = "some other stuff goes here"
        task_spec = TaskSpec(version, problem_type, discount_factor,
                             observations, actions, rewards, extra)

        self.assertEqual(str(task_spec), task_spec_str)

    def test_task_spec_string_names(self):
        task_spec_str = "VERSION RL-Glue-3.0 PROBLEMTYPE episodic " \
            "DISCOUNTFACTOR 1 OBSERVATIONS INTS (0 1) (0 1) (0 1) " \
            "DOUBLES (-1.200000 0.500000) (-1.200000 0.500000) (-0.070000 0.070000) CHARCOUNT 2 " \
            "ACTIONS INTS (0 4) REWARDS (-5.0 5.0) " \
            "EXTRA OBSERVATIONS INTS oi1 oi2 oi3 DOUBLES od1 od2 od3 " \
            "CHARS oc1 oc2 ACTIONS INTS ai1"
        version = 'RL-Glue-3.0'
        problem_type = 'episodic'
        discount_factor = 1
        observations = Space({'oi1': Integer(0, 1), 'oi2': Integer(0, 1),
                              'oi3': Integer(0, 1),
                              'od1': Double(-1.2, 0.5),
                              'od2': Double(-1.2, 0.5),
                              'od3': Double(-0.07, 0.07),
                              'oc1': Char(), 'oc2': Char()})
        actions = Space({'ai1': Integer(0, 4)})
        rewards = Double(-5.0, 5.0)
        extra = "OBSERVATIONS INTS oi1 oi2 oi3 DOUBLES od1 od2 od3 " \
            "CHARS oc1 oc2 ACTIONS INTS ai1"
        task_spec = TaskSpec(version, problem_type, discount_factor,
                             observations, actions, rewards, extra)

        self.assertEqual(task_spec.version, version)
        self.assertEqual(task_spec.problem_type, problem_type)
        self.assertEqual(task_spec.discount_factor, discount_factor)
        self.assertEqual(task_spec.observations, observations)
        self.assertEqual(task_spec.actions, actions)
        self.assertEqual(task_spec.rewards, rewards)
        self.assertEqual(task_spec.extra, extra)
        self.assertEqual(str(task_spec), task_spec_str)

    def test_task_spec_string_actions(self):
        task_spec_str = "VERSION RL-Glue-3.0 PROBLEMTYPE episodic " \
            "DISCOUNTFACTOR 1 OBSERVATIONS INTS (0 1) (0 1) (0 1) " \
            "DOUBLES (-1.200000 0.500000) (-1.200000 0.500000) (-0.070000 0.070000) CHARCOUNT 2 " \
            "ACTIONS INTS (0 4) DOUBLES (0.000000 1.000000) CHARCOUNT 2 " \
            "REWARDS (-5.0 5.0) " \
            "EXTRA OBSERVATIONS INTS oi1 oi2 oi3 DOUBLES od1 od2 od3 " \
            "CHARS oc1 oc2 ACTIONS INTS ai1 DOUBLES ad1 CHARS ac1 ac2 "
        version = 'RL-Glue-3.0'
        problem_type = 'episodic'
        discount_factor = 1
        observations = Space({'oi1': Integer(0, 1), 'oi2': Integer(0, 1),
                              'oi3': Integer(0, 1),
                              'od1': Double(-1.2, 0.5),
                              'od2': Double(-1.2, 0.5),
                              'od3': Double(-0.07, 0.07),
                              'oc1': Char(), 'oc2': Char()})
        actions = Space({'ai1': Integer(0, 4), 'ad1': Double(0.0, 1.0),
                         'ac1': Char(), 'ac2': Char()})
        rewards = Double(-5.0, 5.0)
        extra = "OBSERVATIONS INTS oi1 oi2 oi3 DOUBLES od1 od2 od3 " \
            "CHARS oc1 oc2 ACTIONS INTS ai1 DOUBLES ad1 CHARS ac1 ac2 "
        task_spec = TaskSpec(version, problem_type, discount_factor,
                             observations, actions, rewards, extra)

        self.assertEqual(task_spec.version, version)
        self.assertEqual(task_spec.problem_type, problem_type)
        self.assertEqual(task_spec.discount_factor, discount_factor)
        self.assertEqual(task_spec.observations, observations)
        self.assertEqual(task_spec.actions, actions)
        self.assertEqual(task_spec.rewards, rewards)
        self.assertEqual(task_spec.extra, extra)
        self.assertEqual(str(task_spec), task_spec_str)

    def test_task_spec_order(self):
        task_spec_str = "VERSION RL-Glue-3.0 PROBLEMTYPE episodic " \
            "DISCOUNTFACTOR 1 OBSERVATIONS INTS (0 1) (0 1) (0 1) " \
            "DOUBLES (-1.2 0.5) (-1.2 0.5) (-0.07 0.07) CHARCOUNT 2 " \
            "ACTIONS INTS (0 4) REWARDS (-5.0 5.0) " \
            "EXTRA OBSERVATIONS INTS oi3 oi1 oi2 DOUBLES od3 od1 od2 " \
            "CHARS oc1 oc2 ACTIONS INTS ai2"
        version = 'RL-Glue-3.0'
        problem_type = 'episodic'
        discount_factor = 1
        observations = Space({'oi1': Integer(0, 1), 'oi2': Integer(0, 1),
                              'oi3': Integer(0, 1),
                              'od1': Double(-1.2, 0.5),
                              'od3': Double(-1.2, 0.5),
                              'od2': Double(-0.07, 0.07),
                              'oc1': Char(), 'oc2': Char()})
        actions = Space({'ai2': Integer(0, 4)})
        rewards = Double(-5.0, 5.0)
        extra = "OBSERVATIONS INTS oi3 oi1 oi2 DOUBLES od3 od1 od2 " \
            "CHARS oc1 oc2 ACTIONS INTS ai2"
        task_spec = TaskSpec.parse(task_spec_str)

        self.assertEqual(task_spec.version, version)
        self.assertEqual(task_spec.problem_type, problem_type)
        self.assertEqual(task_spec.discount_factor, discount_factor)
        self.assertEqual(task_spec.observations, observations)
        self.assertEqual(task_spec.actions, actions)
        self.assertEqual(task_spec.rewards, rewards)
        self.assertEqual(task_spec.extra, extra)

    def test_task_spec_equal(self):
        task_spec_str = "VERSION RL-Glue-3.0 PROBLEMTYPE episodic " \
            "DISCOUNTFACTOR 1 OBSERVATIONS INTS (3 0 1) " \
            "DOUBLES (2 -1.2 0.5) (-.07 .07) CHARCOUNT 0 " \
            "ACTIONS INTS (0 4)  REWARDS (-5.0 5.0) " \
            "EXTRA some other stuff goes here"
        task_spec = TaskSpec.parse(task_spec_str)
        other = TaskSpec.parse(task_spec_str)
        self.assertEqual(task_spec, other)


class TestMessageHandler(unittest.TestCase):
    def setUp(self):
        class TestHandler(MessageHandler):
            def on_greet(self, who='World'):
                return 'Hello, %s!' % who
        self.handler = TestHandler()

    def test_message(self):
        message = simplejson.dumps({'function_name': 'greet',
                                    'args': (), 'kwargs': {}})
        answer = self.handler.message(message)
        self.assertEqual(simplejson.loads(answer), "Hello, World!")

    def test_message_args(self):
        message = simplejson.dumps({'function_name': 'greet',
                                    'args': ('User',), 'kwargs': {}})
        answer = self.handler.message(message)
        self.assertEqual(simplejson.loads(answer), "Hello, User!")

    def test_message_malformed(self):
        message = simplejson.dumps({'hola': 'mundo'})
        self.assertRaises(ValueError, self.handler.message, message)


if __name__ == '__main__':
    unittest.main()
