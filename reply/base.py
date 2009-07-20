class _nodefault(object): pass

class Parameter(object):
    def __init__(self, docstring, default=_nodefault):
        self.__doc__ = docstring
        self.default = default

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
