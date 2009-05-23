from rlglue.utils.TaskSpecVRLGLUE3 import TaskSpecParser

from reply.types import Char, Double, Integer, Space

class TaskSpec(object):

    def __init__(self, version='RL-Glue-3.0', problem_type=None,
                 discount_factor=1.0,
                 observations=None, actions=None, rewards=None,
                 extra=None):
        self.version = version
        self.problem_type = problem_type
        self.discount_factor = discount_factor
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.extra = extra

    def __str__(self):
#        observation_names = self.observations.names
#        observation_values = self.observations.values
#        action_names = self.actions.names
#        action_values = self.actions.values
#        extra = ("%s OBSERVATION_NAMES %s ACTION_NAMES %s" %
#                 (self.extra, observation_names, action_names))

        # build observations string
        integers = self.observations[Integer]
        doubles = self.observations[Double]
        charcount = len(self.observations[Char])
        observations_str = ""
        if integers:
            observations_str += "INTS"
            for value in integers:
                observations_str += " (%s %s)" % (value.min, value.max)
        if doubles:
            observations_str += " DOUBLES"
            for value in doubles:
                observations_str += " (%s %s)" % (value.min, value.max)
        if charcount:
            observations_str += " CHARCOUNT %s" % charcount

        # build actions string
        actions_str = ""
        integers = self.actions[Integer]
        doubles = self.actions[Double]
        charcount = len(self.actions[Char])
        actions_str = ""
        if integers:
            actions_str += "INTS"
            for value in integers:
                actions_str += " (%s %s)" % (value.min, value.max)
        if doubles:
            actions_str += " DOUBLES"
            for value in doubles:
                actions_str += " (%s %s)" % (value.min, value.max)
        if charcount:
            actions_str += " CHARCOUNT %s" % charcount

        # build complete string
        return ("VERSION %s " % self.version +
                "PROBLEMTYPE %s " % self.problem_type +
                "DISCOUNTFACTOR %s " % self.discount_factor +
                "OBSERVATIONS %s " % observations_str +
                "ACTIONS %s " % actions_str +
                "REWARDS (%s %s) " % (self.rewards.min, self.rewards.max) +
                "EXTRA %s" % self.extra)

    @classmethod
    def parse(cls, string):
        task_spec = TaskSpec()

        parser = TaskSpecParser(string)
        task_spec.version = parser.getVersion()
        task_spec.problem_type = parser.getProblemType()
        task_spec.discount_factor = parser.getDiscountFactor()

        # parse observations
        int_observations = [Integer(*values) for values in
                            parser.getIntObservations()]
        double_observations = [Double(*values) for values in
                               parser.getDoubleObservations()]
        charcount_observations = [Char()] * parser.getCharCountObservations()
        task_spec.observations = {Integer: int_observations,
                                  Double: double_observations,
                                  Char: charcount_observations}

        # parse actions
        int_actions = [Integer(*values) for values in parser.getIntActions()]
        double_actions = [Double(*values) for values in
                          parser.getDoubleActions()]
        charcount_actions = [Char()] * parser.getCharCountActions()
        task_spec.actions = {Integer: int_actions,
                             Double: double_actions,
                             Char: charcount_actions}

        # parse rewards
        rewards = map(float, parser.getReward()[1:-1].split())
        task_spec.rewards = Double(*rewards)

        task_spec.extra = parser.getExtra()
        return task_spec


class MessageHandler(object):
    def message(self, in_message):
        out_message = ''
        message = simplejson.loads(in_message)
        try:
            fname = message['function_name']
            f = getattr(self, "on_"+fname, None)
            if f is not None:
                out_message = f(*message['args'], **f['kwargs'])
        except:
            print "WARNING: received a malformed message in", self
        return simplejson.dumps(out_message)

