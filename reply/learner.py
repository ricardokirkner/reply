"""Learner classes."""

class Learner(object):        

    """Learner base class."""
    def __init__(self):
        # variable holding a reference to the agent
        self.rl = None

    def new_episode(self):
        """Start a new episode.

        Histories and traces should be cleared here. This should
        also propagate the event to the storage.
        """
        pass
        
    def update(self, state, action, reward, next_state):
        """Update the (state, action, next_state) -> reward relationship.
        
        The parameters are received in rl-encoding.
        """
        raise NotImplementedError()
        
        
class QLearner(Learner):

    """Learner class implemeting the Q algorithm."""

    def __init__(self, alpha, gamma, alpha_decay = 1, min_alpha=None):
        """Initialize the larner.

        Arguments:
        alpha -- learning rate
        gamma -- discount rate

        Keyword arguments:
        alpha_decay -- learning rate decay
        min_alpha   -- minimum learning rate

        """
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.min_alpha = min_alpha
        self.gamma = gamma
        super(QLearner).__init__(self)
        
    def new_episode(self):
        """Start a new episode."""
        super(QLearner, self).new_episode()
        self.alpha *= self.alpha_decay
        if self.min_alpha is not None:
            self.alpha = max(self.min_alpha, self.alpha)
        
    def update(self, state, action, reward, next_state):
        """Update the (state, action, next_state) -> reward relationship."""
        prev_value = self.rl.storage.get_value(state, action)
        max_value_next = self.rl.storage.get_max_value( next_state )
        
        new_value = (
            prev_value + self.alpha *  
            ( reward + self.gamma*max_value_next - prev_value )       
            )
          
        #print "%f + %f * ( %f + %f * %f - %f )"%(
        #    prev_value, self.alpha, reward, self.gamma, max_value_next, 
        #    prev_value)
        
        #print "state:", state, prev_value, "->", next_state, new_value, 
        #print "(r=%i, a=%i)"%(reward, action)
        #print "max_next", max_value_next
        self.rl.storage.store_value(state, action, new_value)
            

class SarsaLearner(QLearner):       

    """Learner class implemeting the Sarsa algorithm."""

    def update(self, state, action, reward, next_state):
        """Update the (state, action, next_state) -> reward relationship."""
        prev_value = self.rl.storage.get_value(state, action)
        next_action = self.rl.selector.select_action(
             next_state 
            )
        max_value_next = self.rl.storage.get_value(next_state, next_action)
        
        new_value = (
            prev_value + self.alpha *  
            ( reward + self.gamma*max_value_next - prev_value )       
            )
          
        
        #print "state:", state, prev_value, "->", new_value, 
        #print "(r=%i, a=%i)"%(reward, action)
        #print "max_next", max_value_next
        self.rl.storage.store_value(state, action, new_value)
