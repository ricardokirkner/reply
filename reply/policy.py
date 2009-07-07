import numpy
import random


class Policy(object):
    def __init__(self, storage):
        self.storage = storage

    def __eq__(self, other):
        return self.storage == other.storage

    def select_action(self, observation):
        raise NotImplementedError()

    def get_mappings(self):
        raise NotImplementedError()


class EGreedyPolicy(Policy):
    def __init__(self, storage, random_action_rate=0.0,
                 random_action_rate_decay=1.0, random_action_rate_min=0.0):
        super(EGreedyPolicy, self).__init__(storage)
        self.random_action_rate = random_action_rate
        self.random_action_rate_decay = random_action_rate_decay
        self.random_action_rate_min = random_action_rate_min

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
            action_id = numpy.argmax(action_values)
        action = self.storage.get_action(action_id)
        return action

    def get_mappings(self):
        actions = []
        for state in self.storage.get_states():
            encoded_action = self.storage.filter(state, numpy.argmax)
            action = self.storage.encoder.encoder['action'].decode(
                (encoded_action,))
            actions.append((state, action))
        return actions


class SoftMaxPolicy(Policy):
    def __init__(self, storage, temperature=0.0):
        super(SoftMaxPolicy, self).__init__(storage)
        self.temperature = temperature

    def __eq__(self, other):
        return (super(SoftMaxPolicy, self).__eq__(other) and
                self.temperature == other.temperature)

    def select_action(self, observation):
        action_values = self.storage.get(observation)
        if self.temperature == 0:
            # this should be absolute greedy selection
            action_id = numpy.argmax(action_values)
        else:
            # get all actions for this state, and their values
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
        action = self.storage.get_action(action_id)
        return action

    def get_mappings(self):
        actions = []
        for state in self.storage.get_states():
            encoded_action = self.storage.filter(state, numpy.argmax)
            action = self.storage.encoder.encoder['action'].decode(
                (encoded_action,))
            actions.append((state, action))
        return actions
