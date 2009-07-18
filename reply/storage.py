"""Storage classes."""
import numpy

from reply.datatypes import Space
from reply.encoder import DummyEncoder, SpaceEncoder


class Storage(object):

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

    def __init__(self, observations=None, actions=None,
                 observation_encoder=None, action_encoder=None):
        super(TableStorage, self).__init__()
        if observation_encoder is None:
            observation_encoder = SpaceEncoder(observations)
        if action_encoder is None:
            action_encoder = SpaceEncoder(actions)
        self.observation_encoder = observation_encoder
        self.action_encoder = action_encoder
        size = observation_encoder.space.size + action_encoder.space.size
        self.data = numpy.zeros(size)

    def __eq__(self, other):
        return (self.observation_encoder == other.observation_encoder and
                self.action_encoder == other.action_encoder and
                (self.data == other.data).all())

    def get(self, observation, action=None):
        """
        >>> from reply.datatypes import Integer, Space
        >>> observations = Space({'o': Integer(0, 0)})
        >>> actions = Space({'a': Integer(0, 0)})
        >>> storage = TableStorage(observations, actions)
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
        encoded_observation = self.observation_encoder.encode(observation)
        if action is not None:
            encoded_action = self.action_encoder.encode(action)
            value = self.data[encoded_observation, encoded_action]
        else:
            value = self.data[encoded_observation]
        return value

    def set(self, observation, action, value):
        """
        >>> from reply.datatypes import Integer, Space
        >>> observations = Space({'o': Integer(0, 0)})
        >>> actions = Space({'a': Integer(0, 0)})
        >>> storage = TableStorage(observations, actions)
        >>> observation = {'o': 0}
        >>> action = {'a': 0}
        >>> storage.get(observation, action)
        0.0
        >>> storage.set(observation, action, 5)
        >>> storage.get(observation, action)
        5.0
        """
        encoded_observation = self.observation_encoder.encode(observation)
        encoded_action = self.action_encoder.encode(action)
        self.data[encoded_observation, encoded_action] = value

    def clear(self):
        """
        >>> from reply.datatypes import Integer, Space
        >>> observations = Space({'o': Integer(0, 0)})
        >>> actions = Space({'a': Integer(0, 0)})
        >>> storage = TableStorage(observations, actions)
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

    def filter(self, observation, action=None, filter=None):
        """
        >>> from reply.datatypes import Integer, Space
        >>> observations = Space({'o': Integer(0, 0)})
        >>> action_encoder = Space({'a': Integer(0, 9)})
        >>> storage = TableStorage(observations, actions)
        >>> observation = {'o': 0}
        >>> for item in range(10):
        ...     storage.set(observation, {'a': item}, item)
        >>> storage.filter(observation, filter=max)
        9.0
        """
        if action is None:
            encoded_values = self.get(observation)
        else:
            encoded_values = self.get(observation, action)
        if filter is not None:
            filtered_values = filter(encoded_values)
        else:
            filtered_values = encoded_values
        return filtered_values

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

