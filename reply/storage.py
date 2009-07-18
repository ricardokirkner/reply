"""Storage classes."""
import numpy

from reply.datatypes import Space
from reply.encoder import DummyEncoder, SpaceEncoder, StateActionEncoder
from reply.base import AgentComponent, Parameter

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

    def filter(self, item, filter):
        raise NotImplementedError()


class TableStorage(Storage):

    """Storage that uses a table for its data."""

    def __init__(self, state_encoder=None, action_encoder=None):
        """
        >>> from reply.datatypes import Integer, Space
        >>> from reply.encoder import SpaceEncoder
        >>> state_encoder = SpaceEncoder(Space({'o': Integer(0, 0)}))
        >>> action_encoder = SpaceEncoder(Space({'a': Integer(0, 0)}))
        >>> storage = TableStorage(state_encoder, action_encoder)
        >>> storage.data
        array([[ 0.]])
        """
        super(TableStorage, self).__init__()
        self.encoder = StateActionEncoder(state_encoder, action_encoder)
        size = (state_encoder.space.size, action_encoder.space.size)
        self.data = numpy.zeros(size)

    def __eq__(self, other):
        return (self.encoder == other.encoder and
                (self.data == other.data).all())

    def get(self, item):
        """
        >>> from reply.datatypes import Integer, Space
        >>> from reply.encoder import SpaceEncoder
        >>> state_encoder = SpaceEncoder(Space({'o': Integer(0, 0)}))
        >>> action_encoder = SpaceEncoder(Space({'a': Integer(0, 0)}))
        >>> storage = TableStorage(state_encoder, action_encoder)
        >>> observation = {'o': 0}
        >>> action = {'a': 0}
        >>> storage.get((observation, action))
        0.0
        >>> action['a'] = 1
        >>> storage.get((observation, action))
        Traceback (most recent call last):
          File "<console>", line 1, in <module>
          File "storage.py", line 80, in get
            return self.data[item]
        IndexError: index (1) out of range (0<=index<1) in dimension 1
        """
        encoded_item = self.encoder.encode(item)
        value = self.data[encoded_item]
        return value

    def set(self, item, value):
        """
        >>> from reply.datatypes import Integer, Space
        >>> from reply.encoder import SpaceEncoder
        >>> state_encoder = SpaceEncoder(Space({'o': Integer(0, 0)}))
        >>> action_encoder = SpaceEncoder(Space({'a': Integer(0, 0)}))
        >>> storage = TableStorage(state_encoder, action_encoder)
        >>> observation = {'o': 0}
        >>> action = {'a': 0}
        >>> storage.get((observation, action))
        0.0
        >>> storage.set((observation, action), 5)
        >>> storage.get((observation, action))
        5.0
        """
        encoded_item = self.encoder.encode(item)
        self.data[encoded_item] = value

    def clear(self):
        """
        >>> from reply.datatypes import Integer, Space
        >>> from reply.encoder import SpaceEncoder
        >>> state_encoder = SpaceEncoder(Space({'o': Integer(0, 0)}))
        >>> action_encoder = SpaceEncoder(Space({'a': Integer(0, 0)}))
        >>> storage = TableStorage(state_encoder, action_encoder)
        >>> observation = {'o': 0}
        >>> action = {'a': 0}
        >>> storage.get((observation, action))
        0.0
        >>> storage.set((observation, action), 1)
        >>> storage.get((observation, action))
        1.0
        >>> storage.clear()
        >>> storage.get((observation, action))
        0.0
        """
        self.data = numpy.zeros(self.data.shape)

    def filter(self, item, filter):
        """
        >>> from reply.datatypes import Integer, Space
        >>> from reply.encoder import SpaceEncoder
        >>> state_encoder = SpaceEncoder(Space({'o': Integer(0, 0)}))
        >>> action_encoder = SpaceEncoder(Space({'a': Integer(0, 9)}))
        >>> storage = TableStorage(state_encoder, action_encoder)
        >>> observation = {'o': 0}
        >>> for item in range(10):
        ...     storage.set((observation, {'a': item}), item)
        >>> storage.filter((observation,), max)
        9.0
        """
        encoded_values = self.get(item)
        filtered_values = filter(encoded_values)
        return filtered_values

    def get_states(self):
        state_space = self.encoder.encoder['state'].space
        for value in state_space.get_items():
            yield value

    def get_actions(self):
        action_space = self.encoder.encoder['action'].space
        for value in action_space.get_items():
            yield value

    def get_action(self, encoded_action):
        if not isinstance(encoded_action, (tuple, list)):
            encoded_action = (encoded_action,)
        action = self.encoder.encoder['action'].decode(encoded_action)
        return action
