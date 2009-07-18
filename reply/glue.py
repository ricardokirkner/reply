
try:
    import rlglue
    has_rlglue = True
except ImportError:
    has_rlglue = False

if has_rlglue:
    from rlglue.RLGlue import RL_init, RL_start, RL_step, RL_cleanup, \
        RL_agent_message, RL_env_message
    from rlglue.agent.Agent import Agent
    from rlglue.agent import AgentLoader
    from rlglue.environment.Environment import Environment
    from rlglue.environment import EnvironmentLoader
    from rlglue.types import Action, Observation, Reward_observation_terminal
else:
    from reply.mock.rlglue import RL_init, RL_start, RL_step, RL_cleanup, \
        RL_agent_message, RL_env_message
    from reply.mock.rlglue import Agent
    from reply.mock.rlglue import AgentLoader
    from reply.mock.rlglue import Environment
    from reply.mock.rlglue import EnvironmentLoader
    from reply.mock.rlglue import Action, Observation, \
        Reward_observation_terminal
from reply.util import TaskSpec
from reply.datatypes import Integer, Double, Char


__all__ = ["start_agent", "start_environment"]


class RlGlueProxyAgent(Agent):
    """This class provides the interface needed by agents
    to be compatible with the rlglue API.

    It acts like an adapter between a reply Agent and a rlglue Agent.
    """
    def __init__(self, agent):
        # agent is the reply Agent instance
        self.agent = agent

    def agent_init(self, task_spec_str):
        task_spec = TaskSpec.parse(task_spec_str)
        self.agent.init(task_spec)

    def agent_start(self, observation):
        observation_space = self.agent.model.observations
        action_space = self.agent.model.actions
        reply_observation = adapt(observation, observation_space)
        reply_action = self.agent.start(reply_observation)
        action = adapt(reply_action, action_space, Action)
        return action

    def agent_step(self, reward, observation):
        observation_space = self.agent.model.observations
        action_space = self.agent.model.actions
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
        observation_space = self.environment.model.observations
        reply_observation = self.environment.start()
        observation = adapt(reply_observation, observation_space, Observation)
        return observation

    def env_step(self, action):
        action_space = self.environment.model.actions
        reply_action = adapt(action, action_space)
        result = self.environment.step(reply_action)
        observation_space = self.environment.model.observations
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
        experiment.set_glue_experiment(self)

    def init(self):
        RL_init()

    def start(self):
        RL_start()

    def step(self):
        roat = RL_step()
        # XXX do an adapt here when we have full taskspec support
        reply_roat = dict(terminal=roat.terminal)
        return reply_roat

    def cleanup(self):
        RL_cleanup()

    def agent_call(self, function_name, *args, **kwargs):
        return self.agent_message(simplejson.dumps(dict(
            function_name=function_name,
            args=args, kwargs=kwargs)))

    def agent_message(self, message):
        RL_agent_message(message)

    def env_call(self, function_name, *args, **kwargs):
        return self.env_message(simplejson.dumps(dict(
            function_name=function_name,
            args=args, kwargs=kwargs)))

    def env_message(self, message):
        RL_env_message(message)

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
                for name in names:
                    if name in source:
                        _array.append(source[name])
        elif target in (Reward_observation_terminal,):
            result.r = source.get('reward', 0.0)
            result.o = adapt(source, space, Observation)
            result.terminal = source.get('terminal', False)
    else:
        # adapt from type to dictionary

        # get attribute names (in order)
        ints = space.get_names_list(Integer)
        doubles = space.get_names_list(Double)
        chars = space.get_names_list(Char)

        # build result dictionary
        result = {'': []}
        if len(source.intArray):
            for i, value in enumerate(source.intArray):
                try:
                    key = ints[i]
                    result[key] = value
                except IndexError:
                    result[''].append(value)
        if len(source.doubleArray):
            for i, value in enumerate(source.doubleArray):
                try:
                    key = doubles[i]
                    result[key] = value
                except IndexError:
                    result[''].append(value)
        if len(source.charArray):
            for i, value in enumerate(source.charArray):
                try:
                    key = chars[i]
                    result[key] = value
                except IndexError:
                    result[''].append(value)

    return result

def start_agent(agent):
    rlglue_agent = RlGlueProxyAgent(agent)
    AgentLoader.loadAgent(rlglue_agent)

def start_environment(env):
    rlglue_environment = RlGlueProxyEnvironment(env)
    EnvironmentLoader.loadEnvironment(rlglue_environment)

def start_experiment(experiment):
    rlglue_experiment = RlGlueProxyExperiment(experiment)
    experiment.run()
