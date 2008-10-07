import numpy
import random


class Selector(object):
    def __init__(self, storage=None):
        self._storage = storage

    @apply
    def storage():
        """
        Storage getter/setter. A selector has to be bound to a storage
        as it doesn't make sense otherwise.
        """
        def fget(self):
            if self._storage is not None:
                return self._storage
            else:
                raise ValueError('Storage has not yet been set.')

        def fset(self, value):
            self._storage = value

        return property(**locals())

    def new_episode(self):
        """
        This function is called whenever a new episode is started.
        Histories and traces should be cleared here.
        """
        pass

    def select_action(self, action_value_array):
        """
        Implements an action selection procedure. The input parameter is
        an array of the values corresponding to the actions.
        The parameters are received in rl-encoding
        """
        raise NotImplementedError()


class EGreedySelector(Selector):
    def __init__(self, epsilon, decay=1, min_epsilon=0, storage=None):
        super(EGreedySelector, self).__init__(storage=storage)
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon

    def new_episode(self):
        self.epsilon = max(self.min_epsilon, self.epsilon*self.decay)

    def select_action(self, encoded_state):
        action_value_array = self.storage.get_state_values(encoded_state)
        if random.random() < self.epsilon:
            #print "R",
            action = random.randint(0, numpy.size(action_value_array)-1)
        else:
            #print action_value_array,
            action = numpy.argmax(action_value_array)
        #print action
        return action


class SoftMaxSelector(Selector):
    def __init__(self, temperature, storage=None):
        super(SoftMaxSelector, self).__init__(storage=storage)
        self.temperature = temperature

    def select_action(self, encoded_state):
        action_value_array = self.storage.get_state_values(encoded_state)
        if self.temperature == 0:
            # this should be absolute greedy selection
            action = numpy.argmax(action_value_array)
        else:
            # get all actions for this state, and their values
            # select a probability
            pr = random.random()
            # get the total value
            temperature = self.temperature
            total_value = sum([numpy.exp(value/temperature) 
                              for value in action_value_array])
            # select the softmax action
            current_pr = 0
            for action,value in enumerate(action_value_array):
                # get the action probability
                action_pr = numpy.exp(value/temperature) / total_value
                # total all past action probabilities
                current_pr += action_pr
                if pr < current_pr:
                    break
        return action

