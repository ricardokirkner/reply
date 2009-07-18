from decimal import Decimal

class Encoder(object):
    def encode(self, item):
        raise NotImplementedError()

    def decode(self, item):
        raise NotImplementedError()


class DummyEncoder(Encoder):
    def encode(self, item):
        return item
    def decode(self, item):
        return item


class BucketEncoder(Encoder):
    def __init__(self, param, num_buckets=2):
        self._min = param.min
        self._max = param.max
        self._num_buckets = num_buckets
        self.buckets = self._generate_buckets()

    @apply
    def min():
        def fget(self):
            return self._min
        def fset(self, value):
            self._min = value
            self.buckets = self._generate_buckets()
        return property(fget, fset)

    @apply
    def max():
        def fget(self):
            return self._max
        def fset(self, value):
            self._max = value
            self.buckets = self._generate_buckets()
        return property(fget, fset)

    @apply
    def num_buckets():
        def fget(self):
            return self._num_buckets
        def fset(self, value):
            self._num_buckets = value
            self.buckets = self._generate_buckets()
        return property(fget, fset)

    def _generate_buckets(self):
        step = Decimal(str((self.max-self.min)/float(self.num_buckets)))
        next = self.min
        buckets = []
        while next <= self.max:
            buckets.append(next)
            next += step
        return buckets

    def encode(self, item):
        if item > self.max or item < self.min:
            raise ValueError("item (%f) outside of encoding range" % item)

        item = Decimal(str(item))
        for bucket, bucket_start in enumerate(self.buckets[:-1]):
            if item >= bucket_start and item < self.buckets[bucket+1]:
                break
        return bucket

    def decode(self, encoded_item):
        if encoded_item < 0 or encoded_item > self.num_buckets-1:
            raise ValueError("item (%d) is not a valid encoding" % encoded_item)

        bucket_start = self.buckets[encoded_item]
        return float(bucket_start)


class SpaceEncoder(Encoder):
    def __init__(self, space):
        self.space = space

    def __eq__(self, other):
        return self.space == other.space

    def encode(self, item):
        """
        >>> from reply.datatypes import Integer, Space
        >>> space = Space({'choice': Integer(0, 9)})
        >>> encoder = SpaceEncoder(space)
        >>> encoder.encode({'choice': 5})
        (5,)
        """
        encoded_item = [item[key] - self.space[key].min \
                        for key in self.space.get_names_list()]
        return tuple(encoded_item)

    def decode(self, encoded_item):
        """
        >>> from reply.datatypes import Integer, Space
        >>> space = Space({'choice': Integer(0, 9)})
        >>> encoder = SpaceEncoder(space)
        >>> encoder.decode((5,))
        {'choice': 5}
        """
        item = {}
        for key, value in zip(self.space.get_names_list(), encoded_item):
            if value is not None:
                if value >= self.space[key].size:
                    raise ValueError("Encoded value (%s) outside of valid range: [%s, %s]."
                        % (value, self.space[key].max - self.space[key].min))
                item[key] = value + self.space[key].min
            else:
                item[key] = value
        return item

    def get_action(self, action_id):
        """
        >>> from reply.datatypes import Integer, Space
        >>> space = Space({'choice': Integer(0, 9)})
        >>> encoder = SpaceEncoder(space)
        >>> encoder.get_action(2)
        {'choice': 2}
        """
        return self.decode((action_id,))


class CompoundSpaceEncoder(SpaceEncoder):
    def __init__(self, space, encoders=None):
        super(CompoundSpaceEncoder, self).__init__(space)
        self.encoders = encoders if encoders is not None else {}

    def encode(self, item):
        encoded_item = []
        for key in self.space.get_names_list():
            if key in self.encoders:
                encoded_key = self.encoders[key].encode(item[key])
            else:
                encoded_key = item[key]
            encoded_item.append(encoded_key)
        return tuple(encoded_item)

    def decode(self, encoded_item):
        item = {}
        for key, encoded_value in zip(self.space.get_names_list(), encoded_item):
            if key in self.encoders:
                value = self.encoders[key].decode(encoded_value)
            else:
                value = encoded_value
            item[key] = value
        return item

