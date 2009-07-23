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


class BucketStorage(Storage):
    model = Parameter("The (observations, actions) model")

    storage_observation_buckets = Parameter("the number of buckets to "\
                                            "divide each dimension", {})
    storage_action_buckets = Parameter("the number of buckets to "\
                                            "divide each dimension", {})

    def __init__(self, agent):
        super(BucketStorage, self).__init__(agent)
        okeys = sorted(self.model.observations.get_names_list())
        akeys = sorted(self.model.actions.get_names_list())

        osizes = [self.storage_observation_buckets.get(k,
                    self.model.observations[k].max -
                    self.model.observations[k].min + 1)
                for k in okeys]
        asizes = [self.storage_action_buckets.get(k,
                    self.model.actions[k].max -
                    self.model.actions[k].min + 1)
                for k in akeys]
        oranges = [self.model.observations[k].max -
                    self.model.observations[k].min
                for k in okeys]
        aranges = [self.model.actions[k].max -
                    self.model.actions[k].min
                for k in akeys]
        ooffsets = [ self.model.observations[k].min
                for k in okeys]
        aoffsets = [ self.model.actions[k].min
                for k in akeys]
        self.data = numpy.zeros(osizes + asizes)
        self.sizes = osizes + asizes
        self.action_sizes = asizes
        self.action_keys = akeys
        self.action_ranges = aranges
        self.action_offsets = aoffsets
        self.ranges = oranges + aranges
        self.offsets = ooffsets + aoffsets
        self.keys = okeys + akeys

    def __eq__(self, other):
        return (self.osizes == other.osizes and
                self.asizes == other.asizes and
                self.okeys == other.okeys and
                self.akeys == other.akeys and
                (self.data == other.data).all())

    def encode(self, observation, action=None):
        action = action if action else {}
        item = [ y for (x,y) in
                    sorted(observation.items()) + sorted(action.items()) ]
        encoded =  [ min(self.sizes[i]-1,      # put in [0, size)
                int(                    # int buckets
                (item[i]- self.offsets[i]+0.0001) # moved to 0
                / self.ranges[i] # ranged into [0,1]
                * self.sizes[i])) # distribute among buckets
                for i in range(len(item))
               ]
        return tuple(encoded)


    def get(self, observation, action=None):
        key = self.encode(observation, action)
        return self.data[key]


    def set(self, observation, action, value):
        key = self.encode(observation, action)
        #print "SET", key, "-->", value
        self.data[key] = value

    def clear(self):
        self.data = numpy.zeros(self.data.shape)

    def get_max_value(self, observation):
        key = self.encode(observation)
        return numpy.max(self.data[key])

    def get_actions(self):
        result = []
        combinations = product(*[ xrange(s) for s in self.action_sizes ])
        combinations =  list(combinations)
        for c in combinations:
            r = []
            for i, v in enumerate(c):
                v = float(v)
                nv = v/self.action_sizes[i]*self.action_ranges[i] \
                        + self.action_offsets[i]
                r.append(nv)
            result.append( dict(zip(self.action_keys, r)))
        return result


    def get_action(self, encoded_action):
        return dict([
            (k,
                (float(encoded_action[i])/self.action_sizes[i]*self.action_ranges[i]
                 + self.action_offsets[i])
            )
            for i, k in enumerate(self.action_keys)])


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
        item = self.observations_mapping.value(observation)
        if action is not None:
            item += self.actions_mapping.value(action)
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
        item = self.observations_mapping.value(observation)
        item += self.actions_mapping.value(action)
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
        values = self.get(observation)
        image_action_id = values.argmax()
        image_shape = []
        for item in self.actions_mapping.image.get_values():
            image_shape.append(item.max - item.min + 1)
        image_action = numpy.unravel_index(image_action_id, image_shape)
        max_action = self.actions_mapping.value(image_action, inverse=True)
        return max_action

    def get_observations(self):
        observation_space = self.observations_mapping.domain
        for value in observation_space.get_items():
            yield value

    def get_actions(self):
        action_space = self.actions_mapping.domain
        for value in action_space.get_items():
            yield value

    def get_action(self, action_image):
        if not isinstance(action_image, (tuple, list)):
            action_image = (action_image,)
        action = self.actions_mapping.value(action_image, inverse=True)
        return action
