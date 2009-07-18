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

    def __eq__(self, other):
        return self.policy == other.policy


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
        prev_value = self.policy.storage.get(state, action)
        if next_state is None:
            max_value_next = 0
        else:
            max_value_next = self.policy.storage.get_max_value(next_state)

        new_value = (
            prev_value + self.learning_rate *
            ( reward + self.value_discount*max_value_next - prev_value )
            )

        self.policy.storage.set(state, action, new_value)

    def __eq__(self, other):
        return (super(QLearner, self).__eq__(other) and
                self.learning_rate == other.learning_rate and
                self.learning_rate_decay == other.learning_rate_decay and
                self.learning_rate_min == other.learning_rate_min and
                self.value_discount == other.value_discount)


class SarsaLearner(QLearner):

    """Learner implementing the Sarsa algorithm."""

    def update(self, state, action, reward, next_state):
        """Update the (state, action, next_state) -> reward relationship."""
        prev_value = self.policy.storage.get(state, action)
        if next_state is None:
            max_value_next = 0
        else:
            next_action = self.policy.select_action(next_state)
            max_value_next = self.policy.storage.get(next_state, next_action)

        new_value = (
            prev_value + self.learning_rate *
            ( reward + self.value_discount*max_value_next - prev_value )
            )

        self.policy.storage.set(state, action, new_value)
