
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
        self.agent.init(task_spec)

    def agent_start(self, observation):
        observation_space = self.agent._observation_space
        action_space = self.agent._action_space
        reply_observation = adapt(observation, observation_space)
        reply_action = self.agent.start(reply_observation)
        action = adapt(reply_action, action_space, Action)
        return action

    def agent_step(self, reward, observation):
        observation_space = self.agent._observation_space
        action_space = self.agent._action_space
        reply_observation = adapt(observation, observation_space)
        reply_action = self.agent.step(reward, reply_observation)
        action = adapt(reply_action, action_space, Action)
        return action

    def agent_end(self, reward):
        self.agent.end(reward)

    def agent_cleanup(self):
        self.agent.cleanup()

    def agent_message(self, in_message):
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
        task_spec = self.environment.init()
        return str(task_spec)

    def env_start(self):
        observation_space = self.environment._observation_space
        reply_observation = self.environment.start()
        observation = adapt(reply_observation, observation_space, Observation)
        return observation

    def env_step(self, action):
        action_space = self.environment._action_space
        reply_action = adapt(action, action_space)
        result = self.environment.step(reply_action)
        observation_space = self.environment._observation_space
        rot = adapt(result, observation_space, Reward_observation_terminal)
        return rot

    def env_cleanup(self):
        self.environment.cleanup()

    def env_message(self, in_message):
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
        roat = RL_step()
        return roat

    def cleanup(self):
        RL_cleanup()

    def run(self):
        steps = 0
        terminal = False
        while steps < 100 and not terminal:
            roat = self.step()
            terminal = roat.terminal
            reward = roat.r
            observation = roat.o
            action = roat.a
            print 'terminal, reward, observation, action', terminal, reward, str(observation), str(action)
            steps += 1


def adapt(source, space, target=None):
    if target is not None:
        # adapt from dictionary to type
        result = target()
        if target in (Action, Observation):
            for _type in (Integer, Double, Char):
                names = space.get_names_list(_type)
                if _type == Integer:
                    _array = result.intArray
                elif _type == Double:
                    _array = result.doubleArray
                elif _type == Char:
                    _array = result.charArray
                else:
                    continue
                for name in names:
                    if name in source:
                        _array.append(source[name])
        elif target in (Reward_observation_terminal,):
            result.o = adapt(source, space, Observation)
            result.terminal = source.get('terminal', False)
            result.reward = source.get('reward', 0)
    else:
        # adapt from type to dictionary

        # get attribute names (in order)
        ints = space.get_names_list(Integer)
        doubles = space.get_names_list(Double)
        chars = space.get_names_list(Char)

        # build result dictionary
        result = {}
        if source.intArray:
            for i, value in enumerate(source.intArray):
                key = ints[i]
                result[key] = value
        if source.doubleArray:
            for i, value in enumerate(source.doubleArray):
                key = doubles[i]
                result[key] = value
        if source.charArray:
            for i, value in enumerate(source.charArray):
                key = chars[i]
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


