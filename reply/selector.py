"""Selector classes."""
import numpy
import random


class Selector(object):

    """Selector base class."""

    def __init__(self, rl):
        """Initialize the selector.
        
        Arguments:
        rl -- an instance of the agent
        
        """
        self.rl = rl

    def new_episode(self):
        """Start a new episode.
        
        Histories and traces should be cleared here.

        """
        pass

    def select_action(self, action_value_array):
        """Return an action.

        Implements an action selection procedure. The input parameter is
        an array of the values corresponding to the actions.
        The parameters are received in rl-encoding
        
        """
        raise NotImplementedError()


class EGreedySelector(Selector):

    """Epsilon-greedy selector."""

    def __init__(self, rl):
        """Initialize the selector.

        Arguments:
        rl -- an instance of the agent

        """
        super(EGreedySelector, self).__init__(rl)
        self.random_action_rate = getattr(rl, "random_action_rate", 0.)
        self.random_action_rate_decay = getattr(
            rl, "random_action_rate_decay", 1.)
        self.random_action_rate_min = getattr(
            rl, "random_action_rate_min", 0.)

    def new_episode(self):
        """Start a new episode."""
        self.random_action_rate = max(
            self.random_action_rate_min,
            self.random_action_rate*self.random_action_rate_decay)

    def end_episode(self):
        """End the current episode."""
        self.rl.current_episode.random_action_rate = self.random_action_rate
        
    def select_action(self, state):
        """Return an action.

        The action returned is the optimal action with probability (1-epsilon),
        and a random action with probability epsilon.

        """
        action_value_array = self.rl.storage.get_state_values(state)
        if random.random() < self.random_action_rate:
            action = random.randint(0, numpy.size(action_value_array)-1)
        else:
            action = numpy.argmax(action_value_array)
        return action


class SoftMaxSelector(Selector):

    """Softmax selector."""

    def __init__(self, rl):
        """Initialize the selector.

        Arguments:
        rl -- an instance of the agent

        """
        super(SoftMaxSelector, self).__init__(rl)
        self.temperature = getattr(rl, "temperature", 0.)

    def end_episode(self):
        """End the current episode."""
        self.rl.current_episode.temperature = self.temperature
        
    def select_action(self, state):
        """Return an action.

        If the selector's temperature is 0, it behaves exactly as a
        completely greedy selector. If the selector's temperature is 1, it
        behaves exactly as a random action selector. Otherwise, the action
        returned is the optimal action according to the softmax selection
        policy.

        """
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
            for action, value in enumerate(action_value_array):
                # get the action probability
                action_pr = numpy.exp(value/temperature) / total_value
                # total all past action probabilities
                current_pr += action_pr
                if pr < current_pr:
                    break
        return action

