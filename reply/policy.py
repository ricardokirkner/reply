import numpy
import random

from reply.base import AgentComponent, Parameter

class Policy(AgentComponent):

    """A Policy represents a mapping between observations and actions."""

    def __eq__(self, other):
        """Return True if both policies are equal."""
        return self.agent.storage == other.agent.storage

    def on_episode_start(self):
        """Handle the start of an episode."""
        pass

    def select_action(self, observation):
        """Return the action associated to the *observation*.

        Raise a NotImplementedError, as this method is supposed to be 
        overriden.

        """
        raise NotImplementedError()

    def get_mappings(self):
        """Return a list of (observation, action) items.

        These items represent the policy.

        """
        actions = []
        for observation in self.agent.storage.get_observations():
            print "test", observation
            action = self.agent.storage.get_max_action(observation)
            actions.append((observation, action))
        return actions


class EGreedyPolicy(Policy):

    """A policy that returns the mapped action with a probability epsilon,
    and returns a random action with a probability (1-epsilon).

    Parameters:

    - random_action_rate -- the chance of taking a random action (epsilon)

    - random_action_rate_decay -- the decay rate between episodes

    - random_action_rate_min -- the mininum value for the random action rate

    """

    random_action_rate = Parameter("The chance of taking a random action", 0.0)
    random_action_rate_decay = Parameter("the random action rate decay", 1.0)
    random_action_rate_min = Parameter("The minimum random action rate", 0.0)

    def on_episode_start(self):
        """Handle an episode start.

        Adjust the random action rate, according to its decay and
        minimum values.

        """
        self.random_action_rate = max(self.random_action_rate_min,
                                      self.random_action_rate_decay *
                                      self.random_action_rate)

    def __eq__(self, other):
        """Return True if both policies are equal."""
        return (super(EGreedyPolicy, self).__eq__(other) and
               self.random_action_rate == other.random_action_rate and
                self.random_action_rate_decay == \
                    other.random_action_rate_decay and
                self.random_action_rate_min == other.random_action_rate_min)

    def select_action(self, observation):
        """Return the action associated to the *observation*."""
        actions = self.agent.storage.get_actions()
        if random.random() < self.random_action_rate:
            action = random.choice(actions)
        else:
            action_values = [self.agent.storage.get(observation, action) for action in actions]
            action_id = numpy.argmax(action_values)
            action = actions[action_id]
        return action


class GreedyPolicy(EGreedyPolicy):

    """A fully greedy policy that always returns the mapped action."""

    random_action_rate = 0
    random_action_rate_decay = 0
    random_action_rate_min = 0


class SoftMaxPolicy(Policy):

    """A policy that implements the SoftMax algorithm.

    Parameters:

    - temperature -- the softmax temperature value

    """

    temperature = Parameter("The softmax temperature", 0.0)

    def __eq__(self, other):
        """Return True if both policies are equal."""
        return (super(SoftMaxPolicy, self).__eq__(other) and
                self.temperature == other.temperature)

    def select_action(self, observation):
        """Return the mapped action for the *observation*."""
        actions = self.agent.storage.get_actions()
        action_values = [self.agent.storage.get(observation, action) for action in actions]
        print action_values
        if self.temperature == 0:
            # this should be absolute greedy selection
            action_id = numpy.argmax(action_values)
        else:
            # get all actions for this observation, and their values
            # select a probability
            pr = random.random()
            # get the total value
            temperature = self.temperature
            total_value = sum([numpy.exp(value/temperature)
                              for value in action_values])
            # select the softmax action
            current_pr = 0
            for action_id, value in enumerate(action_values):
                # get the action probability
                action_id_pr = numpy.exp(value/temperature) / total_value
                # total all past action probabilities
                current_pr += action_id_pr
                if pr < current_pr:
                    break
        action = actions[action_id]
        return action
