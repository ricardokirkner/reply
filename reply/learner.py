
class Learner(object):        
    def __init__(self, policy=None):
        self._policy = policy

    @apply
    def policy():
        def fget(self):
            if self._policy is not None:
                return self._policy
            else:
                raise ValueError('Policy has not yet been set.')

        def fset(self, value):
            self._policy = value

        return property(**locals())

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
    def __init__(self, learning_rate, value_discount, learning_rate_decay = 1, min_learning_rate=None, policy=None):
        """
        implements Q-Learning
        learning_rate: learning rate
        value_discount: value discount
        """
        super(QLearner, self).__init__(policy=policy)
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.min_learning_rate = min_learning_rate
        self.value_discount = value_discount
        
    def new_episode(self):
        super(QLearner, self).new_episode()
        self.learning_rate *= self.learning_rate_decay
        if self.min_learning_rate is not None:
            self.learning_rate = max(self.min_learning_rate, self.learning_rate)
        
    def update(self, state, action, reward, next_state):
        policy = self.policy
        prev_value = policy.get_value(state, action)
        max_value_next = policy.get_max_value(next_state)
        
        new_value = (
            prev_value + self.learning_rate *  
            ( reward + self.value_discount*max_value_next - prev_value )       
            )
          
        #print "%f + %f * ( %f + %f * %f - %f )"%(
        #    prev_value, self.learning_rate, reward, self.value_discount, max_value_next, prev_value)
        
        #print "state:", state, prev_value, "->", next_state, new_value, 
        #print "(r=%i, a=%i)"%(reward, action)
        #print "max_next", max_value_next
        policy.update(state, action, new_value)
            
class SarsaLearner(QLearner):       
    def update(self, state, action, reward, next_state):
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
