
class Parameter(object):
    pass


class Number(Parameter):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __eq__(self, other):
        return type(self) == type(other) and \
            (self.min == other.min and self.max == other.max)


class Integer(Number):
    def __str__(self):
        return "(%i %i)" % (self.min, self.max)


class Double(Number):
    def __str__(self):
        return "(%f %f)" % (self.min, self.max)


class Char(Parameter):

    def __eq__(self, other):
        return True


class Space(object):
    def __init__(self, spec=None, order=None):
        if spec is None:
            self.spec = {}
        else:
            self.spec = spec
        self.order = order
        self._data = {Integer: {}, Double: {}, Char: {}}

        self._build_data()

    def __getitem__(self, item):
        return self._data[item]

    def __str__(self):
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

    def __eq__(self, other):
        return self.spec == other.spec

    def __iter__(self):
        return iter(self._data)

    def get_names_spec(self):
        names = []
        for _type in (Integer, Double, Char):
            if self._data[_type]:
                names.append('INTS')
            elif self._data[_type]:
                names.append('DOUBLES')
            elif self._data[_type]:
                names.append('CHARS')
            else:
                continue
            names.extend(self.get_names_list(_type))
        return ' '.join(names)

    def get_names_list(self, type=None):
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

    def _build_data(self):
        spec = self.spec
        self._data.update(spec)
        for name, value in spec.iteritems():
            _type = type(value)
            values = self._data.setdefault(_type, {})
            values[name] = value

