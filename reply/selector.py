import numpy
import random

class Selector(object):
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
    def __init__(self, epsilon, decay=1):
        self.epsilon = epsilon
        self.decay = decay
        
    def new_episode(self):
        self.epsilon *= self.decay
        
    def select_action(self, action_value_array):
        if random.random() < self.epsilon:
            #print "R",
            action = random.randint(0, numpy.size(action_value_array)-1)
        else:
            #print action_value_array,
            action = numpy.argmax( action_value_array )
        #print action
        return action
    