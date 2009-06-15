"""Learner classes."""

class Learner(object):

    """Learner base class."""

    def __init__(self, policy):
        self.policy = policy

    def new_episode(self):
        pass

    def update(self, state, action, reward, next_state):
        """Update the (state, action, next_state) -> reward relationship.

        next_state is None when state is a final state.
        """
        raise NotImplementedError()


class QLearner(Learner):

    """Learner implementing the Q algorithm."""

    def __init__(self, policy, learning_rate, learning_rate_decay=1.0,
                 learning_rate_min=0.0, value_discount=1.0):
        super(QLearner, self).__init__(policy)
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_min = learning_rate_min
        self.value_discount = value_discount

    def new_episode(self):
        """Start a new episode."""
        super(QLearner, self).new_episode()
        self.learning_rate *= self.learning_rate_decay
        self.learning_rate = max(self.learning_rate_min, self.learning_rate)
#        self.rl.current_episode.learning_rate = self.learning_rate

    def update(self, state, action, reward, next_state):
        """Update the (state, action, next_state) -> reward relationship."""
        prev_value = self.policy.storage.get((state, action))
        if next_state is None:
            max_value_next = 0
        else:
            max_value_next = self.policy.storage.filter(next_state, max)

        new_value = (
            prev_value + self.learning_rate *
            ( reward + self.value_discount*max_value_next - prev_value )
            )

        self.policy.storage.set((state, action), new_value)


#class SarsaLearner(QLearner):
#
#    """Learner implementing the Sarsa algorithm."""
#
#    def update(self, state, action, reward, next_state):
#        """Update the (state, action, next_state) -> reward relationship."""
#        policy = self.policy
#        prev_value = policy.get_value(state, action)
#        next_action = policy.get_action(next_state)
#        max_value_next = policy.get_value(next_state, next_action)
#
#        new_value = (
#            prev_value + self.learning_rate *
#            ( reward + self.value_discount*max_value_next - prev_value )
#            )
#
#
#        #print "state:", state, prev_value, "->", new_value,
#        #print "(r=%i, a=%i)"%(reward, action)
#        #print "max_next", max_value_next
#        policy.update(state, action, new_value)
