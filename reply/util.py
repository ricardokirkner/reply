from reply.types import Space

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
        observation_names = self.observations.names
        observation_values = self.observations.values
        action_names = self.actions.names
        action_values = self.actions.values
        extra = ("%s OBSERVATION_NAMES %s ACTION_NAMES %s" %
                 (self.extra, observation_names, action_names))
        return ("VERSION %s " % self.version +
                "PROBLEMTYPE %s " % self.problem_type +
                "DISCOUNTFACTOR %s " % self.discount_factor +
                "OBSERVATIONS %s " % observation_values +
                "ACTIONS %s " % action_values +
                "REWARDS (%s %s) " % self.rewards +
                "EXTRA %s " % extra)

    @classmethod
    def parse(cls, string):
        task_spec = TaskSpec()

        parts = string.split(' ')
        for i, part in enumerate(parts):
            if part == 'VERSION':
                version = parts[i+1].split('-')[2]
                task_spec.version = float(version)
            elif part == 'PROBLEMTYPE':
                problem_type = parts[i+1]
                task_spec.problem_type = problem_type
            elif part == 'DISCOUNTFACTOR':
                discount_factor = parts[i+1]
                task_spec.discount_factor = float(discount_factor)
            elif part == 'OBSERVATIONS':
                j = i + 1
                observations = []
                while j < len(parts) and parts[j] != 'ACTIONS':
                    observations.append(parts[j])
                    j += 1
                observations = Space.parse(observations)
                task_spec.observations = observations
            elif part == 'ACTIONS':
                j = i + 1
                actions = []
                while j < len(parts) and parts[j] != 'REWARDS':
                    actions.append(parts[j])
                    j += 1
                actions = Space.parse(actions)
                task_spec.actions = actions
            elif part == 'REWARDS':
                j = i + 1
                rewards = parts[i+1] + ', ' + parts[i+2]
                rewards = eval(rewards)
                task_spec.rewards = rewards
            elif part == 'EXTRA':
                extra = parts[i+1:]
                task_spec.extra = extra

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

