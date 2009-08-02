import cPickle as pickle
import traceback

class _nodefault(object): pass

class Parameter(object):

    """A Parameter represents a parameter's abstraction.

    It allows for automatically determining the availability of required
    parameters, and their default values by means of introspection.

    """

    def __init__(self, docstring, default=_nodefault):
        self.__doc__ = docstring
        self.default = default

    def __call__(self, *args, **kwargs):
        raise Exception("Parameter objects cannot be directly called; "
            "they have to be replaced by the expected parameter instance.")

class AgentComponent(object):

    """An AgentComponent represents an exchangable part within an agent."""

    def __init__(self, agent):
        """Initialize the AgentComponent instance.

        Initialization has two primary effects:

        - Binds the component to the agent

        - Validates that all required parameters are provided

        """
        self.agent = agent
        for name in dir(self):
            value = getattr(self, name, None)
            if isinstance(value, Parameter):
                try:
                    new_value = getattr(agent, name)
                except AttributeError, e:
                    if value.default == _nodefault:
                        raise AttributeError(
                            "Configuration of %s requires that the agent %s "\
                            "has an attribute %s: %s" %(
                                self.__class__.__name__,
                                agent.__class__.__name__,
                                name, value.__doc__))
                    else:
                        new_value = value.default
                setattr(self, name, new_value)


class PersistingObject(object):

    """This is a mixin class providing persistence capabilities."""

    def load(self, prefix=""):
        """Load an object from a file and return it.

        The filename used for retrieving the pickled object is determined
        by the object's class name. If present, *prefix* is prepended to the
        object's class name for defining the filename.

        """
        if not prefix:
            prefix = self.__class__.__name__
        filename = "%s.dump" % prefix.lower()
        try:
            f = open(filename, 'rb')
            agent = pickle.load(f)
            f.close()
        except Exception:
            print "%s could not be loaded. Continuing" % prefix
            traceback.print_exc()
            agent = self
        return agent

    def save(self, prefix=""):
        """Save an object to a file.

        The filename used for storing the pickled object is determined
        by the object's class name. If present, *prefix* is prepended to the
        object's class name for defining the filename.

        """
        if not prefix:
            prefix = self.__class__.__name__
        filename = "%s.dump" % prefix.lower()
        f = open(filename, 'wb')
        pickle.dump(self, f)
        f.close()


