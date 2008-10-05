
class Learner(object):        
    def new_episode(self):
        """
        This function is called whenever a new episode is started.
        Histories and traces should be cleared here. This should
        also propagate the event to the storage.
        """
        pass
        
    def update(self, state, action, reward, next_state):
        """
        The parameters are received in rl-encoding
        """
        raise NotImplementedError()
        
        
class QLearner(Learner):
    def __init__(self, alpha, gamma, alpha_decay = 1, min_alpha=None):
        """
        implements Q-Learning
        alpha: learning rate
        gamma: value discount
        """
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.min_alpha = min_alpha
        self.gamma = gamma
        
    def new_episode(self):
        super(QLearner, self).new_episode()
        self.alpha *= self.alpha_decay
        if self.min_alpha is not None:
            self.alpha = max(self.min_alpha, self.alpha)
        
    def update(self, policy, state, action, reward, next_state):
        policy = self.agent.policy
        prev_value = policy.get_value(state, action)
        max_value_next = policy.get_max_value(next_state)
        
        new_value = (
            prev_value + self.alpha *  
            ( reward + self.gamma*max_value_next - prev_value )       
            )
          
        #print "%f + %f * ( %f + %f * %f - %f )"%(
        #    prev_value, self.alpha, reward, self.gamma, max_value_next, prev_value)
        
        #print "state:", state, prev_value, "->", next_state, new_value, 
        #print "(r=%i, a=%i)"%(reward, action)
        #print "max_next", max_value_next
        policy.update(state, action, new_value)
            
class SarsaLearner(QLearner):       
    def update(self, state, action, reward, next_state):
        policy = self.agent.policy
        prev_value = policy.get_value(state, action)
        next_action = policy.get_action(next_state)
        max_value_next = policy.get_value(next_state, next_action)
        
        new_value = (
            prev_value + self.alpha *  
            ( reward + self.gamma*max_value_next - prev_value )       
            )
          
        
        #print "state:", state, prev_value, "->", new_value, 
        #print "(r=%i, a=%i)"%(reward, action)
        #print "max_next", max_value_next
        policy.update(state, action, new_value)
