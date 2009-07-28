import cPickle as pickle
import traceback

class _nodefault(object): pass

class Parameter(object):
    def __init__(self, docstring, default=_nodefault):
        self.__doc__ = docstring
        self.default = default

    def __call__(self, *args, **kwargs):
        raise Exception("Parameter objects cannot be directly called; "
            "they have to be replaced by the expected parameter instance.")

class AgentComponent(object):
    def __init__(self, agent):
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
    def load(self, prefix=""):
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
        if not prefix:
            prefix = self.__class__.__name__
        filename = "%s.dump" % prefix.lower()
        f = open(filename, 'wb')
        pickle.dump(self, f)
        f.close()


