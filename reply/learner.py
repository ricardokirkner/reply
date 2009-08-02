"""Learner classes."""
from reply.base import AgentComponent, Parameter


class Learner(AgentComponent):
    """Learner base class."""

    def on_episode_start(self):
        """Handle an episode start."""
        pass

    def update(self, state, action, reward, next_state):
        """Update the (state, action, next_state) -> reward relationship.

        next_state is None when state is a final state.

        Raise a NotImplementedError, as this method should be overriden.
        """
        raise NotImplementedError()

    def __eq__(self, other):
        """Return True if both agents are equal."""
        return self.agent.policy == other.agent.policy


class QLearner(Learner):

    """Learner implementing the Q algorithm.

    Parmeters:

    - learning_rate -- the agents learning rate

    - learning_rate_decay -- the learning rate decay per episode

    - learning_rate_min -- the minimum value for the learning rate

    - value_discount -- the value discount

    """

    learning_rate = Parameter("The agents learning rate")
    learning_rate_decay = Parameter("The learning rate decay per episode. "\
                                    "(rate = decay * rate)", 1.0)
    learning_rate_min = Parameter("The minimun value for the learning rate", 0.0)
    value_discount = Parameter("The value discount", 1.0)

    def on_episode_start(self):
        """Start a new episode."""
        super(QLearner, self).on_episode_start()
        self.learning_rate *= self.learning_rate_decay
        self.learning_rate = max(self.learning_rate_min, self.learning_rate)

    def update(self, state, action, reward, next_state):
        """Update the (state, action, next_state) -> reward relationship."""
        prev_value = self.agent.storage.get(state, action)
        if next_state is None:
            max_value_next = 0
        else:
            max_value_next = self.agent.storage.get_max_value(next_state)

        new_value = (
            prev_value + self.learning_rate *
            ( reward + self.value_discount*max_value_next - prev_value )
            )
        self.agent.storage.set(state, action, new_value)

    def __eq__(self, other):
        """Return True if both learners are equal."""
        return (super(QLearner, self).__eq__(other) and
                self.learning_rate == other.learning_rate and
                self.learning_rate_decay == other.learning_rate_decay and
                self.learning_rate_min == other.learning_rate_min and
                self.value_discount == other.value_discount)


class SarsaLearner(QLearner):

    """Learner implementing the Sarsa algorithm."""

    def update(self, state, action, reward, next_state):
        """Update the (state, action, next_state) -> reward relationship."""
        prev_value = self.agent.storage.get(state, action)
        if next_state is None:
            max_value_next = 0
        else:
            next_action = self.agent.policy.select_action(next_state)
            max_value_next = self.agent.storage.get(next_state, next_action)

        new_value = (
            prev_value + self.learning_rate *
            ( reward + self.value_discount*max_value_next - prev_value )
            )

        self.agent.storage.set(state, action, new_value)
