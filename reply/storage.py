"""Storage classes."""
import numpy
from itertools import product

from reply.base import AgentComponent, Parameter
from reply.datatypes import Integer, Space
from reply.mapping import OffsetIdentityMapping


class Storage(AgentComponent):

    """A generic storage.

    Storage defines the common API for all storage types.

    """

    model = Parameter("The (observations, actions) model")

    def __init__(self, agent, observations_mapping=None, actions_mapping=None):
        """Construct a new Storage object.

        agent points to the Agent instance using this Storage.

        If present, observations_mapping and actions_mapping must be
        instances of Mapping. Otherwise, they are assumed to be
        OffsetIdentityMapping by default.

        """
        super(Storage, self).__init__(agent)
        if observations_mapping is None:
            observations_mapping = OffsetIdentityMapping(self.model.observations)
        if actions_mapping is None:
            actions_mapping = OffsetIdentityMapping(self.model.actions)
        self.observations_mapping = observations_mapping
        self.actions_mapping = actions_mapping

    def __getitem__(self, item):
        """Return the content associated to the parameter item."""
        return self.get(item)

    def __setitem__(self, item, value):
        """Set the content associated to the parameter item."""
        self.set(item, value)

    def get(self, item):
        """Return the content assocaited to the parameter item."""
        raise NotImplementedError()

    def set(self, item, value):
        """Set the content associated to the parameter item."""
        raise NotImplementedError()

    def clear(self):
        """Reset the storage contents to its default value."""
        raise NotImplementedError()


class TableStorage(Storage):

    """A storage that uses a multi-dimensional table for its contents."""

    def __init__(self, agent, observations_mapping=None, actions_mapping=None):
        """Construct a new TableStorage object.

        Both observations_mapping and actions_mapping must have an image
        space composed entirely by Integer dimensions.

        """
        super(TableStorage, self).__init__(agent,
                                           observations_mapping,
                                           actions_mapping)
        # make sure the output of the mappings is as expected
        observations_image = self.observations_mapping.image
        actions_image = self.actions_mapping.image
        storage_domain_items = observations_image.get_values() + \
                               actions_image.get_values()
        for item in storage_domain_items:
            assert isinstance(item, Integer)

        # build underlying data store
        shape = []
        for item in storage_domain_items:
            item_size = item.max - item.min + 1
            shape.append(item_size)
        self.data = numpy.zeros(shape)
        self.all_actions = [ self.actions_mapping._inverse(item)
                            for item in self.actions_mapping.image.get_items()]
        self.all_observations = [ self.observations_mapping._inverse(item)
                            for item in self.observations_mapping.image.get_items()]

        self.observation_keys = observations_image.get_names_list()
        self.action_keys = actions_image.get_names_list()

    def encode(self, observation, action=None):
        result = [ observation[key] for key in self.observation_keys ]
        if action is not None:
            result += [ action[key] for key in self.action_keys ]
        return tuple(result)

    def decode_action(self, action):
        item = {}
        for i, v in enumerate(action):
            item[self.action_keys[i]] = v
        return item

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
        if action is not None:
            item = self.encode(
                self.observations_mapping.value(observation),
                self.actions_mapping.value(action))
        else:
            item = self.encode(
                self.observations_mapping.value(observation))
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
        item = self.encode(
            self.observations_mapping.value(observation),
            self.actions_mapping.value(action))
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
        values = self.get(observation)
        max_value = values.max()
        return max_value

    def get_max_action(self, observation):
        print "observation", observation
        values = self.get(observation)
        image_action_id = values.argmax()
        image_shape = []
        for item in self.actions_mapping.image.get_values():
            image_shape.append(int(item.max - item.min + 1))
        image_action = numpy.unravel_index(image_action_id, image_shape)
        action = self.decode_action(image_action)
        print "GOT ACTION", action, image_action, image_action_id, values, image_shape
        max_action = self.actions_mapping.value(action, inverse=True)
        return max_action

    def get_observations(self):
        return self.all_observations

    def get_actions(self):
        return self.all_actions
