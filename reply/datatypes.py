import itertools

class Dimension(object):

    """A Dimension represents an attribute that defines a part of a space.

    Both observation and action spaces are described by means of the
    dimensions involved.

    """

    pass


class Number(Dimension):

    """A Number represents a numeric dimension, described by a numeric range.

    Any Number will have *min* and *max* values, describing the range's
    boundaries.

    """

    def __init__(self, min, max):
        """Create a Number.

        Arguments:

        - min -- the (inclusive) lower bound on the range

        - max -- the (inclusive) upper bound on the range

        """
        self.min = min
        self.max = max

    def __eq__(self, other):
        """Return True if both Number instances are equal."""
        return type(self) == type(other) and \
            (self.min == other.min and self.max == other.max)

    def assert_belongs(self, value):
        """Return True if the *value* falls into the Number's range.

        Raise a ValueError otherwise.

        """
        if value > self.max or value < self.min:
            raise ValueError("Image attribute %s outside of range: "
                "[%s, %s] in %s." % (value, self.min, self.max, self))
        return True

class Integer(Number):

    """An Integer is a Number whose values are integer."""

    def __str__(self):
        """Return the string representation of the Integer."""
        return "(%i %i)" % (self.min, self.max)


class Double(Number):

    """A Double is a Number whose values are floats."""

    def __str__(self):
        """Return the string representation of the Double."""
        return "(%f %f)" % (self.min, self.max)


class Char(Dimension):

    """A Char represents a character."""

    def __eq__(self, other):
        """Return True.

        Two Char instances are always equal.

        """
        return True

    def assert_belongs(self, value):
        """Return True if the *value* is a character.

        Raise a ValueError otherwise.

        """
        if not isinstance(value, str) or len(value) != 1:
            raise ValueError("Value %s is not char" % (value,))
        return True


class Model(object):

    """A Model represents the model for a problem.

    Keyword arguments:

    - observations -- the observation Space

    - actions -- the action Space

    """

    def __init__(self, observations=None, actions=None):
        """Create a Model instance."""
        self.observations = self._build_space(observations)
        self.actions = self._build_space(actions)

    def _build_space(self, value):
        if isinstance(value, Space):
            space = value
        elif isinstance(value, dict):
            spec = value.get('spec', None)
            order = value.get('order', None)
            space = Space(spec, order)
        else:
            space = Space()
        return space

    def __eq__(self, other):
        """Return True if both models are equal."""
        return (self.observations == other.observations and
                self.actions == other.actions)


class Space(object):

    """A Space represents a set of Dimensions.

    Keyword arguments:

    - spec -- a dictionary specifying the dimensions for the space

    - order -- a dictionary of (type, list) items representing the space's
               internal ordering

    - valid -- a function used to filter out valid items

    """

    def __init__(self, spec=None, order=None, valid=None):
        """Create a Space instance."""
        if spec is None:
            self.spec = {}
        else:
            self.spec = spec
        self.order = order
        self.valid = valid
        self._data = {Integer: {}, Double: {}, Char: {}}

        self._build_data()

    def __getitem__(self, item):
        """Return the Dimension associated with the *item*."""
        return self._data[item]

    def __getattr__(self, attr):
        """Make the Space behave both as a dictionary and an object.

        Dimensions within a Space can be accessed both as keys and attributes.

        """
        if hasattr(self, 'spec') and hasattr(self, '_data') and \
           attr in self.spec:
            return self._data[attr]
        else:
            raise AttributeError, attr

    def __str__(self):
        """Return the string representation of the Space."""
        items = []
        for item in (Integer, Double, Char):
            if self._data[item]:
                if item == Integer:
                    items.append('INTS')
                    if self.order is not None:
                        for name in self.order[Integer]:
                            items.append(str(self._data[item][name]))
                    else:
                        for name, value in sorted(self._data[item].iteritems()):
                            items.append(str(value))
                elif item == Double:
                    items.append('DOUBLES')
                    if self.order is not None:
                        for name in self.order[Double]:
                            items.append(str(self._data[item][name]))
                    else:
                        for name, value in sorted(self._data[item].iteritems()):
                            items.append(str(value))
                elif item == Char:
                    items.append('CHARCOUNT')
                    items.append(str(len(self._data[item])))
        return ' '.join(items)

    def __repr__(self):
        """Return the internal representation of the Space."""
        return "<Space %s>" % (self.__str__(),)

    def __eq__(self, other):
        """Return True if both Space instances are equal."""
        return (self.spec == other.spec and
                self.order == other.order)

    def __iter__(self):
        """Return an iterator over the Space."""
        return iter(self._data)

    def get_names_spec(self):
        """Return a string specification for the Dimension's names."""
        names = []
        for _type in (Integer, Double, Char):
            if self._data[_type]:
                if _type == Integer:
                    names.append('INTS')
                elif _type == Double:
                    names.append('DOUBLES')
                elif _type == Char:
                    names.append('CHARS')
            else:
                continue
            names.extend(self.get_names_list(_type))
        return ' '.join(names)

    def get_names_list(self, type=None):
        """Return a list of the Dimension's names.

        If *type* is given, only Dimensions matching the type are included.
        Otherwise, all Dimensions are included.

        The results are returned according to the Space's ordering.

        """
        names = []
        if self.order is not None:
            if type is None:
                type_names = self.order.get(Integer, []) + \
                             self.order.get(Double, []) + \
                             self.order.get(Char, [])
            else:
                type_names = self.order.get(type, [])
            for name in type_names:
                names.append(str(name))
        else:
            if type is None:
                type_names = sorted(self._data[Integer].iteritems()) + \
                             sorted(self._data[Double].iteritems()) + \
                             sorted(self._data[Char].iteritems())
            else:
                type_names = sorted(self._data[type].iteritems())
            for name, value in type_names:
                names.append(str(name))
        return names

    def get_items(self):
        """Return a list of dictionaries representing valid items.

        As the Space can represent more values than desired, a filter
        is applied to eliminate unwanted items.

        """
        name_list = self.get_names_list()
        key_values = []
        # build all possible values for each attribute
        for name in name_list:
            name_values = range(self[name].min, self[name].max+1)
            key_values.append(name_values)
        # generate all possible combinations of those values
        values = itertools.product(*key_values)
        # for each combination, create the corresponding item
        items = []
        for value in values:
            item = {}
            for i, name in enumerate(name_list):
                item[name] = value[i]
            items.append(item)
        if self.valid is not None:
            items = filter(self.valid, items)
        return items

    def get_values(self):
        """Return the list of Dimensions defined in the Space."""
        values = []
        for key in self.get_names_list():
            values.append( self[key] )
        return values

    def _build_data(self):
        spec = self.spec
        self._data.update(spec)
        for name, value in spec.iteritems():
            _type = type(value)
            values = self._data.setdefault(_type, {})
            values[name] = value

    def assert_valid(self, point):
        """Return True if a point belongs to this Space.

        Raise a ValueError otherwise.

        """
        if not isinstance(point, dict):
            raise ValueError("Value %s is not a point (dict)" % (point))
        for key, value in point.items():
            self[key].assert_belongs(value)
        return True
