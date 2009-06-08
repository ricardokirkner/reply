"""Storage classes."""
import numpy

from reply.encoder import DummyEncoder


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

    def filter(self, item, filter=None):
        raise NotImplementedError()


class TableStorage(Storage):

    """Storage that uses a table for its data."""

    def __init__(self, size, encoder=None):
        """
        >>> storage = TableStorage((1,1))
        >>> storage.data
        array([[ 0.]])
        """
        super(TableStorage, self).__init__()
        if encoder is None:
           self.encoder = DummyEncoder()
        else:
            self.encoder = encoder
        self.data = numpy.zeros(size)

    def get(self, item, decode=True):
        """
        >>> storage = TableStorage((1, 1))
        >>> storage.get((0,0))
        0.0
        >>> storage.get((0,1))
        Traceback (most recent call last):
          File "<console>", line 1, in <module>
          File "storage.py", line 80, in get
            return self.data[item]
        IndexError: index (1) out of range (0<=index<1) in dimension 1
        """
        encoded_item = self.encoder.encode(item)
        value = self.data[encoded_item]
        if decode:
            value = self.encoder.decode(value)
        return value

    def set(self, item, value):
        """
        >>> storage = TableStorage((1, 1))
        >>> storage.get((0, 0))
        0.0
        >>> storage.set((0, 0), 5)
        >>> storage.get((0, 0))
        5.0
        """
        encoded_item = self.encoder.encode(item)
        self.data[encoded_item] = value

    def clear(self):
        """
        >>> storage = TableStorage((1, 1))
        >>> storage.get((0, 0))
        0.0
        >>> storage.set((0, 0), 1)
        >>> storage.get((0, 0))
        1.0
        >>> storage.clear()
        >>> storage.get((0, 0))
        0.0
        """
        self.data = numpy.zeros(self.data.shape)

    def filter(self, item, filter=None):
        """
        >>> storage = TableStorage((1, 10))
        >>> for item in range(10):
        ...     storage.set((0, item), item)
        >>> storage.filter((0,), max)
        9.0
        """
        encoded_values = self.get(item)
        filtered_values = filter(encoded_values)
        if isinstance(filtered_values, (list, tuple)):
            decoded_values = map(self.encoder.decode, filtered_values)
        else:
            decoded_values = self.encoder.decode(filtered_values)
        return decoded_values


if __name__ == '__main__':
    import doctest
    doctest.testmod()
