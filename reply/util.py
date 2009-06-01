import simplejson

from rlglue.utils.TaskSpecVRLGLUE3 import TaskSpecParser

from reply.types import Char, Double, Integer, Space


def parse_spaces(data, parser):
    observations_spec = {}
    actions_spec = {}
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
            #parse_names(observation_names, observation_values, observations)
            observations_spec = build_spec(observation_names, observation_values)
        elif part == 'ACTIONS':
            action_names = parts[i+1:]
            action_values = {"INTS": parser.getIntActions(),
                             "DOUBLES": parser.getDoubleActions(),
                             "CHARS": parser.getCharCountActions()}
            #parse_names(action_names, action_values, actions)
            actions_spec = build_spec(action_names, action_values)
    observations = Space(observations_spec)
    actions = Space(actions_spec)
    return (observations, actions)

def build_spec(names, values):
    spec = {}
    i = 0
    while i < len(names):
        name = names[i]
        if name == 'INTS':
            i += 1
            for value in values['INTS']:
                name = names[i]
                spec[name] = Integer(*value)
                i += 1
        elif name == 'DOUBLES':
            i += 1
            for value in values['DOUBLES']:
                name = names[i]
                spec[name] = Double(*value)
                i += 1
        elif name == 'CHARS':
            i += 1
            for j in xrange(values['CHARS']):
                name = names[i]
                spec[name] = Char()
                i += 1
    return spec

def parse_names(names, values, result):
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
        #integers = self.observations[Integer]
        #doubles = self.observations[Double]
        #charcount = len(self.observations[Char])
        #observations_str = ""
        #if integers:
        #    observations_str += "INTS"
        #    for value in integers.values():
        #        observations_str += " (%s %s)" % (value.min, value.max)
        #if doubles:
        #    observations_str += " DOUBLES"
        #    for value in doubles.values():
        #        observations_str += " (%s %s)" % (value.min, value.max)
        #if charcount:
        #    observations_str += " CHARCOUNT %s" % charcount
        observations_str = str(self.observations)

        # build actions string
        #integers = self.actions[Integer]
        #doubles = self.actions[Double]
        #charcount = len(self.actions[Char])
        #actions_str = ""
        #if integers:
        #    actions_str += "INTS"
        #    for value in integers.values():
        #        actions_str += " (%s %s)" % (value.min, value.max)
        #if doubles:
        #    actions_str += " DOUBLES"
        #    for value in doubles.values():
        #        actions_str += " (%s %s)" % (value.min, value.max)
        #if charcount:
        #    actions_str += " CHARCOUNT %s" % charcount
        actions_str = str(self.actions)

        # build complete string
        return ("VERSION %s " % self.version +
                "PROBLEMTYPE %s " % self.problem_type +
                "DISCOUNTFACTOR %s " % self.discount_factor +
                "OBSERVATIONS %s " % observations_str +
                "ACTIONS %s " % actions_str +
                "REWARDS (%s %s) " % (self.rewards.min, self.rewards.max) +
                "EXTRA %s" % self.extra)

    def __eq__(self, other):
        return (self.version == other.version and
                self.problem_type == other.problem_type and
                self.discount_factor == other.discount_factor and
                self.observations == other.observations and
                self.actions == other.actions and
                self.rewards == other.rewards and
                self.extra == other.extra)

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
        observations, actions = parse_spaces(task_spec.extra, parser)
        task_spec.observations = observations
        task_spec.actions = actions
        return task_spec


class MessageHandler(object):
    def message(self, in_message):
        out_message = ''
        message = simplejson.loads(in_message)
        try:
            fname = message['function_name']
            f = getattr(self, "on_"+fname, None)
            if f is not None:
                out_message = f(*message['args'], **message['kwargs'])
        except:
            raise ValueError("Received a malformed message in %s: %s" % (self, in_message))
        return simplejson.dumps(out_message)

