
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
            return self._inverse(argument)
        else:
            return self._value(argument)

    def _value(self, argument):
        """Return the image value of the argument."""
        raise NotImplementedError()

    def _inverse(self, argument):
        """Return the domain value of the argument."""
        raise NotImplementedError()


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

        If the argument is not within the range specified by the image, a
        ValueError is raised.

        """
        value = []
        for key in self.domain.get_names_list():
            if argument[key] > self.image[key].max or \
               argument[key] < self.image[key].min:
                raise ValueError("Domain attribute %s value (%s) outside of image range: "
                    "[%s, %s]." % (key, argument[key], self.image[key].min,
                                   self.image[key].max))
            value.append(argument[key])
        return tuple(value)

    def _inverse(self, argument):
        """Return the domain value of the argument.

        If the argument is not within the range specified by the domain, a
        ValueError is raised.

        """
        item = {}
        for key, value in zip(self.domain.get_names_list(), argument):
            if value is not None:
                if value > self.domain[key].max or value < self.domain[key].min:
                    raise ValueError("Image value (%s) outside of domain range: [%s, %s]."
                        % (value, self.domain[key].min, self.domain[key].max))
            item[key] = value
        return item


class OffsetIdentityMapping(IdentityMapping):

    """A Mapping that implements an origin-offset Identity function."""

    def _value(self, argument):
        """Return the image value of the argument.

        If the argument is not within the range specified by the image, a
        ValueError is raised.

        """
        value = []
        for key in self.domain.get_names_list():
            if argument[key] > self.image[key].max or \
               argument[key] < self.image[key].min:
                   raise ValueError("Domain attribute %s value (%s) outside of image range: "
                        "[%s, %s]." % (key, argument[key], self.image[key].min,
                                       self.image[key].max))
            value.append(argument[key] - self.domain[key].min)
        return tuple(value)

    def _inverse(self, argument):
        """Return the domain value of the argument.

        If the argument is not within the range specified by the domain, a
        ValueError is raised.

        """
        item = {}
        for key, value in zip(self.image.get_names_list(), argument):
            if value is not None:
                value += self.domain[key].min
                if value > self.domain[key].max or value < self.domain[key].min:
                    raise ValueError("Image value (%s) outside of domain range: [%s, %s]."
                        % (value, self.domain[key].min, self.domain[key].max))
            item[key] = value
        return item
