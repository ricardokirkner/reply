"""Storage classes."""
import numpy

from reply.base import AgentComponent, Parameter
from reply.datatypes import Space
from reply.encoder import DummyEncoder, SpaceEncoder

class Storage(AgentComponent):

    """Storage base class."""

    def __getitem__(self, item):
        return self.get(item)

    def __setitem__(self, item, value):
        self.set(item, value)

    def get(self, item):
        raise NotImplementedError()

    def set(self, item, value):
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()


class TableStorage(Storage):

    """Storage that uses a table for its data."""

    model = Parameter("The (observations, actions) model")

    def __init__(self, agent):
        super(TableStorage, self).__init__(agent)
        self.observation_encoder = SpaceEncoder(self.model.observations)
        self.action_encoder = SpaceEncoder(self.model.actions)
        size = self.model.observations.size + self.model.actions.size
        self.data = numpy.zeros(size)

    def __eq__(self, other):
        return (self.observation_encoder == other.observation_encoder and
                self.action_encoder == other.action_encoder and
                (self.data == other.data).all())

    def get(self, observation, action=None):
        """
        >>> from reply.datatypes import Integer, Model, Space
        >>> observations = Space({'o': Integer(0, 0)})
        >>> actions = Space({'a': Integer(0, 0)})
        >>> model = Model(observations, actions)
        >>> agent = type('agent', (object,), {'model': model})
        >>> storage = TableStorage(agent)
        >>> observation = {'o': 0}
        >>> action = {'a': 0}
        >>> storage.get(observation, action)
        0.0
        >>> action['a'] = 1
        >>> storage.get(observation, action)
        Traceback (most recent call last):
          File "<console>", line 1, in <module>
          File "storage.py", line 80, in get
            return self.data[item]
        IndexError: index (1) out of range (0<=index<1) in dimension 1
        """
        item = self.observation_encoder.encode(observation)
        if action is not None:
            encoded_action = self.action_encoder.encode(action)
            item += encoded_action
        value = self.data[item]
        return value

    def set(self, observation, action, value):
        """
        >>> from reply.datatypes import Integer, Model, Space
        >>> observations = Space({'o': Integer(0, 0)})
        >>> actions = Space({'a': Integer(0, 0)})
        >>> model = Model(observations, actions)
        >>> agent = type('agent', (object,), {'model': model})
        >>> storage = TableStorage(agent)
        >>> observation = {'o': 0}
        >>> action = {'a': 0}
        >>> storage.get(observation, action)
        0.0
        >>> storage.set(observation, action, 5)
        >>> storage.get(observation, action)
        5.0
        """
        item = self.observation_encoder.encode(observation)
        item += self.action_encoder.encode(action)
        self.data[item] = value

    def clear(self):
        """
        >>> from reply.datatypes import Integer, Model, Space
        >>> observations = Space({'o': Integer(0, 0)})
        >>> actions = Space({'a': Integer(0, 0)})
        >>> model = Model(observations, actions)
        >>> agent = type('agent', (object,), {'model': model})
        >>> storage = TableStorage(agent)
        >>> observation = {'o': 0}
        >>> action = {'a': 0}
        >>> storage.get(observation, action)
        0.0
        >>> storage.set(observation, action, 1)
        >>> storage.get(observation, action)
        1.0
        >>> storage.clear()
        >>> storage.get(observation, action)
        0.0
        """
        self.data = numpy.zeros(self.data.shape)

    def get_max_value(self, observation):
        encoded_values = self.get(observation)
        max_value = encoded_values.max()
        return max_value

    def get_max_action(self, observation):
        encoded_values = self.get(observation)
        encoded_action_id = encoded_values.argmax()
        encoded_action = numpy.unravel_index(encoded_action_id,
                                             self.action_encoder.space.size)
        max_action = self.action_encoder.decode(encoded_action)
        return max_action

    def get_states(self):
        state_space = self.observation_encoder.space
        for value in state_space.get_items():
            yield value

    def get_actions(self):
        action_space = self.action_encoder.space
        for value in action_space.get_items():
            yield value

    def get_action(self, encoded_action):
        if not isinstance(encoded_action, (tuple, list)):
            encoded_action = (encoded_action,)
        action = self.action_encoder.decode(encoded_action)
        return action
