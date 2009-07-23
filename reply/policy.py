import numpy
import random

from reply.base import AgentComponent, Parameter

class Policy(AgentComponent):
    storage = Parameter("Where to store the policy")

    def __eq__(self, other):
        return self.storage == other.storage

    def new_episode(self):
        pass

    def select_action(self, observation):
        raise NotImplementedError()

    def get_mappings(self):
        actions = []
        for observation in self.storage.get_observations():
            action = self.storage.get_max_action(observation)
            actions.append((observation, action))
        return actions


class EGreedyPolicy(Policy):
    random_action_rate = Parameter("The chance of taking a random action", 0.0)
    random_action_rate_decay = Parameter("the random action rate decay", 1.0)
    random_action_rate_min = Parameter("The minimum random action rate", 0.0)

    def new_episode(self):
        self.random_action_rate = max(self.random_action_rate_min,
                                      self.random_action_rate_decay *
                                      self.random_action_rate)

    def __eq__(self, other):
        return (super(EGreedyPolicy, self).__eq__(other) and
                self.random_action_rate == other.random_action_rate and
                self.random_action_rate_decay == \
                    other.random_action_rate_decay and
                self.random_action_rate_min == other.random_action_rate_min)

    def select_action(self, observation):
        action_values = self.storage.get(observation)
        if random.random() < self.random_action_rate:
            action_id = random.randint(0, numpy.size(action_values)-1)
        else:
            #print sorted(observation.items()), self.storage.encode(observation)
            action_id = numpy.argmax(action_values)
            #print action_values, action_id
        actions = list(self.storage.get_actions())
        action = actions[action_id]
        #print "ACTION SELECTED", action_id, action
        return action


class SoftMaxPolicy(Policy):
    temperature = Parameter("The softmax temperature", 0.0)

    def __eq__(self, other):
        return (super(SoftMaxPolicy, self).__eq__(other) and
                self.temperature == other.temperature)

    def select_action(self, observation):
        action_values = self.storage.get(observation)
        if self.temperature == 0:
            # this should be absolute greedy selection
            action_id = numpy.argmax(action_values)
        else:
            # get all actions for this observation, and their values
            # select a probability
            pr = random.random()
            # get the total value
            temperature = self.temperature
            total_value = sum([numpy.exp(value/temperature)
                              for value in action_values])
            # select the softmax action
            current_pr = 0
            for action_id, value in enumerate(action_values):
                # get the action probability
                action_id_pr = numpy.exp(value/temperature) / total_value
                # total all past action probabilities
                current_pr += action_id_pr
                if pr < current_pr:
                    break
        actions = list(self.storage.get_actions())
        action = actions[action_id]
        return action
