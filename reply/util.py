import re
import simplejson

from reply.contrib import argparse

try:
    from rlglue.utils.TaskSpecVRLGLUE3 import TaskSpecParser
except ImportError:
    from reply.contrib.TaskSpecVRLGLUE3 import TaskSpecParser

from reply.datatypes import Char, Double, Integer, Space


def parse_spaces(extra, parser):
    observations_spec = {}
    actions_spec = {}
    parts = extra.split()
    observation_names = []
    action_names = []
    i = 0
    while i < len(parts):
        if parts[i] == 'OBSERVATIONS':
            names = {'INTS': [], 'DOUBLES': [], 'CHARS': []}
            _type = ''
            i += 1
            while i < len(parts) and parts[i] != 'ACTIONS':
                if parts[i] in ('INTS', 'DOUBLES', 'CHARS'):
                    _type = parts[i]
                elif _type:
                    names[_type].append(parts[i])
                i += 1
            observation_names = names
        elif parts[i] == 'ACTIONS':
            names = {'INTS': [], 'DOUBLES': [], 'CHARS': []}
            _type = ''
            i += 1
            while i < len(parts):
                if parts[i] in ('INTS', 'DOUBLES', 'CHARS'):
                    _type = parts[i]
                elif _type:
                    names[_type].append(parts[i])
                i += 1
            action_names = names
        else:
            # skip part
            i += 1

    # observations
    observation_values = {"INTS": parser.getIntObservations(),
                          "DOUBLES": parser.getDoubleObservations(),
                          "CHARS": parser.getCharCountObservations()}
    observations_spec = build_spec(observation_names, observation_values)
    observations = Space(observations_spec)
    # actions
    action_values = {"INTS": parser.getIntActions(),
                     "DOUBLES": parser.getDoubleActions(),
                     "CHARS": parser.getCharCountActions()}
    actions_spec = build_spec(action_names, action_values)
    actions = Space(actions_spec)
    return (observations, actions)

def build_spec(names, values):
    spec = {}
    varcount = 0
    for _type in ('INTS', 'DOUBLES', 'CHARS'):
        i = 0
        if _type == 'INTS':
            int_values = values['INTS']
            if 'INTS' in names:
                int_names = names['INTS']
                # build ints by name
                while i < min(len(int_values), len(int_names)):
                    name = int_names[i]
                    value = int_values[i]
                    spec[name] = Integer(*value)
                    i += 1
            # build ints by value
            while i < len(int_values):
                value = int_values[i]
                spec["var_"+str(varcount)] = Integer(*value)
                varcount += 1
                i += 1
        elif _type == 'DOUBLES':
            double_values = values['DOUBLES']
            if 'DOUBLES' in names:
                double_names = names['DOUBLES']
                # build doubles by name
                while i < min(len(double_values), len(double_names)):
                    name = double_names[i]
                    value = double_values[i]
                    spec[name] = Double(*value)
                    i += 1
            # build doubles by value
            while i < len(double_values):
                value = double_values[i]
                spec["var_"+str(varcount)] = Double(*value)
                varcount += 1
                i += 1
        elif _type == 'CHARS':
            char_values = range(values['CHARS'])
            if 'CHARS' in names:
                char_names = names['CHARS']
                # build chars by name
                while i < min(len(char_values), len(char_names)):
                    name = char_names[i]
                    spec[name] = Char()
                    i += 1
            # build chars by value
            while i < len(char_values):
                value = char_values[i]
                spec["var_"+str(varcount)] = Char()
                varcount += 1
                i += 1
    return spec


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
        observations_str = str(self.observations)
        # build actions string
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
            raise ValueError("Received a malformed message in %s: %s" % (
                self, in_message))
        return simplejson.dumps(out_message)


#########################
### Extending arparse ###
#########################

def boolean(string):
    string = string.lower()
    if string in ['0', 'f', 'false', 'no', 'off']:
        return False
    elif string in ['1', 't', 'true', 'yes', 'on']:
        return True
    else:
        raise ValueError()


class ConfigureAction(argparse.Action):
    def __init__(self, option_strings, dest, default=None,
                 required=False, help=None, metavar=None,
                 positive_prefixes=('--', '--with-', '--enable-'),
                 negative_prefixes=('--no-', '--without-', '--disable-')):
        strings = []
        self.positive_strings = set()
        self.negative_strings = set()
        for string in option_strings:
            assert re.match(r'--[A-z]+', string)
            suffix = string[2:]
            for positive_prefix in positive_prefixes:
                self.positive_strings.add(positive_prefix + suffix)
                strings.append(positive_prefix + suffix)
            for negative_prefix in negative_prefixes:
                self.negative_strings.add(negative_prefix + suffix)
                strings.append(negative_prefix + suffix)
        super(ConfigureAction, self).__init__(option_strings=strings,
            dest=dest, nargs='?', const=None, default=default, type=boolean,
            choices=None, required=required, help=help, metavar=metavar)

    def __call__(self, parser, namespace, value, option_string=None):
        if value is None:
            value = option_string in self.positive_strings
        elif option_string in self.negative_strings:
            value = not value
        setattr(namespace, self.dest, value)
