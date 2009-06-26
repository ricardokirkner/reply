
# Functions

def RL_init():
    pass

def RL_start():
    pass

def RL_step():
    roat = Reward_observation_terminal()
    return roat

def RL_cleanup():
    pass

def RL_agent_message(message):
    pass

def RL_env_message(message):
    pass


# Classes

class Agent(object):
    pass


class AgentLoader(object):
    def loadAgent(self, agent):
        pass


class Environment(object):
    pass


class EnvironmentLoader(object):
    def loadEnvironment(self, environment):
        pass


class Action(object):
    def __init__(self):
        self.intArray = []
        self.doubleArray = []
        self.charArray = []

    def sameAs(self, other):
        return (self.intArray == other.intArray and
                self.doubleArray == other.doubleArray and
                self.charArray == other.charArray)


class Observation(object):
    def __init__(self):
        self.intArray = []
        self.doubleArray = []
        self.charArray = []

    def sameAs(self, other):
        return (self.intArray == other.intArray and
                self.doubleArray == other.doubleArray and
                self.charArray == other.charArray)


class Reward_observation_terminal(object):
    r = 0
    o = Observation()
    terminal = False


class Reward_observation_action_terminal(object):
    r = 0
    o = Observation()
    a = Action()
    terminal = False

