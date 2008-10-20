"""Learner classes."""

class Learner(object):        

    """Learner base class."""

    def __init__(self, rl):
        """Initialize the learner.
        
        Arguments:
        rl -- an instance of the agent.
        
        """
        self.rl = rl

    def new_episode(self):
        """Start a new episode.

        Histories and traces should be cleared here. This should
        also propagate the event to the storage.
        """
        pass
        
    def update(self, state, action, reward, next_state):
        """Update the (state, action, next_state) -> reward relationship.

        The parameters are received in rl-encoding
        
        """
        raise NotImplementedError()
        
        
class QLearner(Learner):

    """Learner implementing the Q algorithm."""

    def __init__(self, rl):
        """Initialize the learner.

        Arguments:
        rl -- an instance of the agent.

        """
        super(QLearner, self).__init__(rl)
        self.learning_rate = getattr(self.rl, "learning_rate")
        self.learning_rate_decay = getattr(self.rl, "learning_rate_decay", 1)
        self.learning_rate_min = getattr(self.rl, "learning_rate_min", 0)
        self.value_discount = getattr(self.rl, "value_discount", 1)
        
    def new_episode(self):
        """Start a new episode."""
        super(QLearner, self).new_episode()
        self.learning_rate *= self.learning_rate_decay
        self.learning_rate = max(self.learning_rate_min, self.learning_rate)
        self.rl.current_episode.learning_rate = self.learning_rate
        
    def update(self, state, action, reward, next_state):
        """Update the (state, action, next_state) -> reward relationship."""
        prev_value = self.rl.storage.get_value(state, action)
        if self.rl.world.is_final():
            max_value_next = 0    
        else:
            max_value_next = self.rl.storage.get_max_value(next_state)
        new_value = (
            prev_value + self.learning_rate *  
            ( reward + self.value_discount*max_value_next - prev_value )       
            )
          
        #print "%f + %f * ( %f + %f * %f - %f )"%(
        #    prev_value, self.learning_rate, reward, self.value_discount,
        #    max_value_next, prev_value)
        
        #print "state:", state, prev_value, "->", next_state, new_value, 
        #print "(r=%i, a=%i)"%(reward, action)
        #print "max_next", max_value_next
        self.rl.storage.store_value(state, action, new_value)

            
class SarsaLearner(QLearner):       

    """Learner implementing the Sarsa algorithm."""

    def update(self, state, action, reward, next_state):
        """Update the (state, action, next_state) -> reward relationship."""
        policy = self.policy
        prev_value = policy.get_value(state, action)
        next_action = policy.get_action(next_state)
        max_value_next = policy.get_value(next_state, next_action)
        
        new_value = (
            prev_value + self.learning_rate *  
            ( reward + self.value_discount*max_value_next - prev_value )       
            )
          
        
        #print "state:", state, prev_value, "->", new_value, 
        #print "(r=%i, a=%i)"%(reward, action)
        #print "max_next", max_value_next
        policy.update(state, action, new_value)
