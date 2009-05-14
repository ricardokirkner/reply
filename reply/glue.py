
from rlglue.RLGlue import RL_init, RL_start, RL_step, RL_cleanup
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader
from rlglue.environment.Environment import Environment
from rlglue.environment import EnvironmentLoader
from rlglue.types import Action, Observation, Reward_observation_terminal

from reply.types import Integer, Double, Char

__all__ = ["start_agent", "start_environment"]


class RlGlueProxyAgent(Agent):
    """This class provides the interface needed by agents
    to be compatible with the rlglue API.

    It acts like an adapter between a reply Agent and a rlglue Agent.
    """
    def __init__(self, agent):
        # agent is the reply Agent instance
        self.agent = agent

    def agent_init(self, task_spec):
        import pdb; pdb.set_trace()
        return self.agent.init(task_spec)

    def agent_start(self, observation):
        import pdb; pdb.set_trace()
        space = self.agent._action_space
        reply_observation = adapt(observation, space)
        reply_action = self.agent.start(observation)
        action = adapt(reply_action, space, Action)
        return action

    def agent_step(self, reward, observation):
        import pdb; pdb.set_trace()
        space = self.agent._action_space
        reply_observation = adapt(observation, space)
        reply_action = self.agent.step(reward, reply_observation)
        action = adapt(reply_action, space, Action)
        return action

    def agent_end(self, reward):
        import pdb; pdb.set_trace()
        self.agent.end(reward)

    def agent_cleanup(self):
        import pdb; pdb.set_trace()
        self.agent.cleanup()

    def agent_message(self, in_message):
        import pdb; pdb.set_trace()
        out_message = self.agent.message(in_message)
        return out_message


class RlGlueProxyEnvironment(Environment):
    """This class provides the interface needed by environments
    to be compatible with the rlglue API.

    It acts like an adapter between a reply Environment and a rlglue
    Environment.
    """
    def __init__(self, environment):
        # agent is the reply Environment instance
        self.environment = environment

    def env_init(self):
        import pdb; pdb.set_trace()
        task_spec = self.environment.init()
        return str(task_spec)

    def env_start(self):
        import pdb; pdb.set_trace()
        space = self.environment._observation_space
        reply_observation = self.environment.start()
        observation = adapt(reply_observation, space, Observation)
        return observation

    def env_step(self, action):
        import pdb; pdb.set_trace()
        space = self.environment._observation_space
        reply_action = adapt(action, space)
        result = self.environment.step(reply_action)
        rot = adapt(result, space, Reward_observation_terminal)
        return rot

    def env_cleanup(self):
        import pdb; pdb.set_trace()
        self.environment.cleanup()

    def env_message(self, in_message):
        import pdb; pdb.set_trace()
        out_message = self.environment.message(in_message)
        return out_message


class RlGlueProxyExperiment(object):
    """This class provides the interface needed by experiments
    to be compatible with the rlglue API.

    It acts like an adapter between a reply Experiment and a rlglue Experiment.
    """
    def __init__(self, experiment):
        # agent is the reply Experiment instance
        self.experiment = experiment
        self._initialized = False
        self._started = False

    def init(self):
        RL_init()
        self.experiment.init()

    def start(self):
        RL_start()
        self.experiment.start()

    def step(self):
        terminal, reward, observation, action = RL_step()
        return terminal, reward, observation, action

    def cleanup(self):
        RL_cleanup()

    def run(self):
        steps = 0
        terminal = False
        while steps < 100 and not terminal:
            terminal, reward, observation, action = self.step()
            print 'terminal, reward, observation, action', terminal, reward, observation, action
            steps += 1


def adapt(source, space, target=None):
    import pdb; pdb.set_trace()
    if target is not None:
        # adapt from dictionary to type
        result = target()
        for key, value in source.items():
            if key not in space.spec.keys():
                raise KeyError("The space does not include a definition for '%s'" % key)
            _type = space.spec[key]
            if isinstance(_type, Integer):
                result.intArray.append(value)
            elif isinstance(_type, Double):
                result.doubleArray.append(value)
            elif isinstance(_type, Char):
                result.charArray.append(value)
            else:
                raise TypeError("%s is of an invalid type: %s" % (key, space.spec[key]))
    else:
        # adapt from type to dictionary
        result = {}
        names = space.names
        for values in (source.intArray, source.doubleArray, source.charArray):
            for i, value in enumerate(values):
                key = keys[i]
                result[key] = value
    return result

def start_agent(agent):
    rlglue_agent = RlGlueProxyAgent(agent)
    AgentLoader.loadAgent(rlglue_agent)

def start_environment(env):
    rlglue_environment = RlGlueProxyEnvironment(env)
    EnvironmentLoader.loadEnvironment(rlglue_environment)

def start_experiment(experiment):
    rlglue_experiment = RlGlueProxyExperiment(experiment)
    rlglue_experiment.init()
    rlglue_experiment.start()
    rlglue_experiment.run()
    rlglue_experiment.cleanup()


