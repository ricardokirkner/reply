import numpy
import random


class Selector(object):
    def __init__(self, rl):
        self.rl = rl

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
    def __init__(self, rl):
        super(EGreedySelector, self).__init__(rl)
        self.random_action_rate = getattr(rl, "random_action_rate", 0.)
        self.random_action_rate_decay = getattr(
            rl, "random_action_rate_decay", 1.)
        self.random_action_rate_min = getattr(
            rl, "random_action_rate_min", 0.)

    def new_episode(self):
        self.random_action_rate = max(
            self.random_action_rate_min,
            self.random_action_rate*self.random_action_rate_decay)

    def end_episode(self):
        self.rl.current_episode.random_action_rate = self.random_action_rate
        
    def select_action(self, state):
        action_value_array = self.rl.storage.get_state_values(state)
        if random.random() < self.random_action_rate:
            action = random.randint(0, numpy.size(action_value_array)-1)
        else:
            action = numpy.argmax(action_value_array)
        return action


class SoftMaxSelector(Selector):
    def __init__(self, rl):
        super(SoftMaxSelector, self).__init__(rl)
        self.temperature = getattr(rl, "temperature", 0.)

    def end_episode(self):
        self.rl.current_episode.temperature = self.temperature
        
    def select_action(self, state):
        action_value_array = self.rl.storage.get_state_values(state)
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

