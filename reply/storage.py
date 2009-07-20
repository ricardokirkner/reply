"""Storage classes."""
import numpy
from itertools import product

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
