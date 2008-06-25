
class Learner(object):
    def __init__(self, storage):
        self.storage = storage
        
    def new_episode(self):
        """
        This function is called whenever a new episode is started.
        Histories and traces should be cleared here. This should
        also propagate the event to the storage.
        """
        self.storage.new_episode()
        
    def update(self, state, action, reward, next_state):
        """
        The parameters are received in rl-encoding
        """
        raise NotImplementedError()
        
class QLearner(Learner):
    def __init__(self, storage, alpha, gamma, alpha_decay = 1, min_alpha=None):
        """
        implements Q-Learning
        alpha: learning rate
        gamma: value discount
        """
        self.storage = storage
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.min_alpha = min_alpha
        self.gamma = gamma
        
    def new_episode(self):
        super(QLearner, self).new_episode()
        self.alpha *= self.alpha_decay
        if self.min_alpha is not None:
            self.alpha = max(self.min_alpha, self.alpha)
        
    def update(self, state, action, reward, next_state):
        prev_value = self.storage.get_value(state, action)
        max_value_next = self.storage.get_max_value( next_state )
        
        new_value = (
            prev_value + self.alpha *  
            ( reward + self.gamma*max_value_next - prev_value )       
            )
          
        
        #print "state:", state, prev_value, "->", new_value, 
        #print "(r=%i, a=%i)"%(reward, action)
        #print "max_next", max_value_next
        self.storage.store_value(state, action, new_value)
        
            