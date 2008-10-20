"""Selector classes."""
import numpy
import random


class Selector(object):

    """Selector base class."""

    def __init__(self):
        # this variable holds a reference to the agent.
        self.rl = None

    def new_episode(self):
        """Start a new episode.

        Histories and traces should be cleared here.
        """
        pass

    def select_action(self, action_value_array):
        """Return an action.

        Implements an action selection procedure. The input parameter is
        an array of the values corresponding to the actions.
        The parameters are received in rl-encoding.
        """
        raise NotImplementedError()


class EGreedySelector(Selector):

    """Epsilon-greedy selector class."""

    def __init__(self, epsilon, decay=1, min_epsilon=0):
        """Initialize the selector.
        
        Arguments:
        epsilon -- random action rate.
        decay   -- random action rate decay.
        min_epsilon -- minimum random action rate.
        """
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
        super(EGreedySelector, self).__init__(self)

    def new_episode(self):
        """Start a new episode."""
        self.epsilon = max(self.min_epsilon, self.epsilon*self.decay)

    def select_action(self, encoded_state):
        """Return an action.
        
        The action returned is the optimal action with probability (1-p),
        and a random action with probability p.
        """
        action_value_array = self.rl.storage.get_state_values( encoded_state )
        if random.random() < self.epsilon:
            #print "R",
            action = random.randint(0, numpy.size(action_value_array)-1)
        else:
            #print action_value_array,
            action = numpy.argmax( action_value_array )
        #print action
        return action


class SoftMaxSelector(Selector):

    """Softmax selector class."""

    def __init__(self, temperature):
        """Initialize the selector.
        
        Arguments:
        temperature -- 
        """
        self.temperature = temperature
        super(SoftMaxSelector, self).__init__(self)

    def select_action(self, encoded_state):
        """Return an action.
        
        If the selector's temperature is 0, then it behaves exactly as
        a completely greedy selector. If the selector's temperature is
        1, then it behaves exactly as a random action selector. Otherwise,
        the action returned is the optimal action according to the
        softmax selection policy.
        """
        action_value_array = self.rl.storage.get_state_values(encoded_state)
        if self.temperature == 0:
            # this should be absolute greedy selection
            action = numpy.argmax(action_value_array)
        else:
            # get all actions for this state, and their values
            # select a probability
            pr = random.random()
            # get the total value
            temperature = self.temperature
            total_value = sum([numpy.exp(value/temperature) \
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

