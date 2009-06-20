import numpy
import random


class Policy(object):
    def __init__(self, storage):
        self.storage = storage

    def select_action(self, observation):
        raise NotImplementedError()

    def __eq__(self, other):
        return self.storage == other.storage


class EGreedyPolicy(Policy):
    def __init__(self, storage, random_action_rate=0.0,
                 random_action_rate_decay=1.0, random_action_rate_min=0.0):
        super(EGreedyPolicy, self).__init__(storage)
        self.random_action_rate = random_action_rate
        self.random_action_rate_decay = random_action_rate_decay
        self.random_action_rate_min = random_action_rate_min

    def select_action(self, observation):
        encoded_actions = self.storage.get(observation)
        if random.random() < self.random_action_rate:
            encoded_action = random.randint(0, numpy.size(encoded_actions)-1)
        else:
            encoded_action = numpy.argmax(encoded_actions)
        decoded_state_action = self.storage.encoder.decode((None,
                                                            encoded_action))
        action = decoded_state_action[1]
        return action

    def __eq__(self, other):
        return (super(EGreedyPolicy, self).__eq__(other) and
                self.random_action_rate == other.random_action_rate and
                self.random_action_rate_decay == \
                    other.random_action_rate_decay and
                self.random_action_rate_min == other.random_action_rate_min)


class SoftMaxPolicy(Policy):
    def __init__(self, storage, temperature=0.0):
        super(SoftMaxPolicy, self).__init__(storage)
        self.temperature = temperature

    def select_action(self, observation):
        encoded_actions = self.storage.get(observation)
        if self.temperature == 0:
            # this should be absolute greedy selection
            encoded_action = numpy.argmax(encoded_actions)
        else:
            # get all actions for this state, and their values
            # select a probability
            pr = random.random()
            # get the total value
            temperature = self.temperature
            total_value = sum([numpy.exp(value/temperature)
                              for value in encoded_actions])
            # select the softmax action
            current_pr = 0
            for encoded_action, value in enumerate(encoded_actions):
                # get the action probability
                encoded_action_pr = numpy.exp(value/temperature) / total_value
                # total all past action probabilities
                current_pr += encoded_action_pr
                if pr < current_pr:
                    break
        decoded_state_action = self.storage.encoder.decode((None,
                                                            encoded_action))
        action = decoded_state_action[1]
        return action

    def __eq__(self, other):
        return (super(SoftMaxPolicy, self).__eq__(other) and
                self.temperature == other.temperature)
