from reply.datatypes import Integer, Double, Space

class Mapping(object):

    """Mapping for transforming one Space into another."""

    def __init__(self, domain, image):
        """Construct a Mapping object.

        Both domain and image arguments are to be Space instances.

        """
        self.domain = domain
        self.image = image

    def value(self, argument, inverse=False):
        """Return the value for the mapping using the parameter argument.

        If inverse is set to False, the mapping is applied from domain to image.
        Otherwise it is applied from image to domain.

        """
        if inverse:
            if __debug__:
                self.image.assert_valid(argument)
            result =  self._inverse(argument)
            if __debug__:
                self.domain.assert_valid(result)
        else:
            if __debug__:
                self.domain.assert_valid(argument)
            result =  self._value(argument)
            if __debug__:
                self.image.assert_valid(result)

        return result
    def _value(self, argument):
        """Return the image value of the argument."""
        raise NotImplementedError()

    def _inverse(self, argument):
        """Return the domain value of the argument."""
        raise NotImplementedError()

    def __repr__(self):
        return "<mapping %s --> %s>" % (self.domain, self.image)


class IdentityMapping(Mapping):

    """A Mapping that implements the Identity function."""

    def __init__(self, domain):
        """Construct an IdentityMapping object.

        Only domain is used as an argument, as both domain and image are
        the same.

        """
        super(IdentityMapping, self).__init__(domain, domain)

    def _value(self, argument):
        """Return the image value of the argument.
        """
        return argument

    def _inverse(self, argument):
        """Return the domain value of the argument.
        """
        return argument


class OffsetIdentityMapping(IdentityMapping):

    """A Mapping that implements an origin-offset Identity function."""
    def __init__(self, domain):
        image = Space(dict([ (key, Integer(0.0, domain[key].max - domain[key].min))
            for key in domain.get_names_list()]))
        super(IdentityMapping, self).__init__(domain, image)

    def _value(self, argument):
        """Return the image value of the argument.
        """
        item = {}
        for key, value in argument.items():
            item[key] = value - self.domain[key].min
        return item

    def _inverse(self, argument):
        """Return the domain value of the argument.
        """
        item = {}
        for key, value in argument.items():
            value += self.domain[key].min
            item[key] = value
        return item

class TileMapping(Mapping):
    def __init__(self, domain, buckets):
        """Maps the domain into a series of buckets per dimension

        buckets should be a dictionary with:
            key = the dimension name
            value = the number of buckets for that dimension
        """
        self.domain = domain
        self.buckets = dict( (k, float(v)) for k,v in buckets.items() )
        for key in domain.get_names_list():
            dim = domain[key]
            if not (isinstance(dim, Integer) or isinstance(dim, Double)):
                raise TypeError("Domain dimensions must be numerical")

        self.image = Space(dict([ (key, Integer(0.0, float(buckets[key])-1))
            for key in domain.get_names_list() ]))
        self.ranges = dict([ (key, float(domain[key].max - domain[key].min))
            for key in domain.get_names_list() ])

    def _value(self, argument):
        item = {}
        for key, value in argument.items():
            item[key] = int(min(self.image[key].max, max(0,
                    (value - self.domain[key].min) /
                    self.ranges[key] * self.buckets[key])))
        return item

    def _inverse(self, argument):
        item = {}
        for key, value in argument.items():
            # report a value in the middle of the range
            half_step = self.ranges[key] / self.buckets[key] / 2
            print "!", self.buckets[key], self.ranges[key]
            print value, value/self.buckets[key],  value / self.buckets[key] * self.ranges[key],  value / self.buckets[key] * self.ranges[key] + self.domain[key].min
            print value / self.buckets[key] * self.ranges[key] + self.domain[key].min + half_step, max(self.domain[key].min,
                    value / self.buckets[key] * self.ranges[key]
                    + self.domain[key].min + half_step), min(self.domain[key].max, max(self.domain[key].min,
                    value / self.buckets[key] * self.ranges[key]
                    + self.domain[key].min + half_step))

            item[key] = min(self.domain[key].max, max(self.domain[key].min,
                    value / self.buckets[key] * self.ranges[key]
                    + self.domain[key].min + half_step))
        return item
