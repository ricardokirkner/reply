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
        # build observations string
        integers = self.observations[Integer]
        doubles = self.observations[Double]
        charcount = len(self.observations[Char])
        observations_str = ""
        if integers:
            observations_str += "INTS"
            for value in integers.values():
                observations_str += " (%s %s)" % (value.min, value.max)
        if doubles:
            observations_str += " DOUBLES"
            for value in doubles.values():
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
            for value in integers.values():
                actions_str += " (%s %s)" % (value.min, value.max)
        if doubles:
            actions_str += " DOUBLES"
            for value in doubles.values():
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
        # parse rewards
        rewards = map(float, parser.getReward()[1:-1].split())
        task_spec.rewards = Double(*rewards)

        task_spec.extra = parser.getExtra()

        # parse observation and action names
        observations, actions = _parse_spaces(task_spec.extra, parser)
        task_spec.observations = observations
        task_spec.actions = actions
        return task_spec


def _parse_spaces(data, parser):
    observations = {Integer: {}, Double: {}, Char: {}}
    actions = {Integer: {}, Double: {}, Char: {}}
    parts = data.split()
    for i, part in enumerate(parts):
        if part == 'OBSERVATIONS':
            try:
                actions_idx = parts.index('ACTIONS')
            except ValueError:
                actions_idx = -1
            observation_names = parts[i+1:actions_idx]
            observation_values = {"INTS": parser.getIntObservations(),
                                  "DOUBLES": parser.getDoubleObservations(),
                                  "CHARS": parser.getCharCountObservations()}
            _parse_names(observation_names, observation_values, observations)
        elif part == 'ACTIONS':
            action_names = parts[i+1:]
            action_values = {"INTS": parser.getIntActions(),
                             "DOUBLES": parser.getDoubleActions(),
                             "CHARS": parser.getCharCountActions()}
            _parse_names(action_names, action_values, actions)
    return (observations, actions)

def _parse_names(names, values, result):
    i = 0
    while i < len(names):
        name = names[i]
        if name == 'INTS':
            data = {}
            i += 1
            for value in values['INTS']:
                name = names[i]
                data[name] = Integer(*value)
                i += 1
            result[Integer] = data
        elif name == 'DOUBLES':
            data = {}
            i += 1
            for value in values['DOUBLES']:
                name = names[i]
                data[name] = Double(*value)
                i += 1
            result[Double] = data
        elif name == 'CHARS':
            data = {}
            i += 1
            for j in xrange(values['CHARS']):
                name = names[i]
                data[name] = Char()
                i += 1
            result[Char] = data


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

