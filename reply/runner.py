from multiprocessing import Process

from reply.contrib import argparse
from reply.glue import start_agent
from reply.glue import start_environment
from reply.glue import start_experiment
from reply.util import ConfigureAction

try:
    import rlglue
    has_rlglue = True
except ImportError:
    has_rlglue = False


class Command(object):
    #: the command name
    name = "replace_me"

    def register(self, parser):
        """Register this command in the parser subparser"""
        pass

    def run(self, agent, env, experiment, args):
        """run the command with options"""
        pass

_commands = []
def register_command(command_klass):
    _commands.append(command_klass)

def unregister_command(command_klass):
    _commands.remove(command_klass)

class Run(Command):
    """Run the experiment in foreground"""
    name = "run"

    def register(self, parser):
        parser.add_argument("--max-episodes", type=int, dest="max_episodes",
                            help="the maximum number of episodes to execute")
        parser.add_argument("--max-steps", type=int, dest="max_steps",
                            help="the maximum number of steps per "
                            "episodes to execute")

        parser.add_argument("--save-agent", action=ConfigureAction,
                            dest="save_agent",
                            help="save the agent to disk on exit")
        parser.add_argument("--load-agent", action=ConfigureAction,
                            dest="load_agent",
                            help="load the agent from disk on start")
        parser.add_argument("--agent-name", type=str, dest="agent_name",
                            help="prefix for the filename to use when "
                            "saving/loading the agent")

        parser.add_argument("--save-storage", action=ConfigureAction,
                            dest="save_storage",
                            help="save the storage to disk on exit")
        parser.add_argument("--load-storage", action=ConfigureAction,
                            dest="load_storage",
                            help="load the storage from disk on start")
        parser.add_argument("--storage-name", type=str, dest="storage_name",
                            help="prefix for the filename to use when "
                            "saving/loading the storage")

    def run(self, agent, env, experiment, args=None):
        self.agent = agent
        self.env = env
        self.experiment = experiment

        self.save_agent = False
        self.load_agent = False
        self.agent_name = ""

        self.save_storage = False
        self.load_storage = False
        self.storage_name = ""

        if args is not None:
            if args.max_episodes is not None:
                experiment.max_episodes = args.max_episodes
            if args.max_steps is not None:
                experiment.max_steps = args.max_steps
            if args.save_agent:
                self.save_agent = args.save_agent
            if args.load_agent:
                self.load_agent = args.load_agent
            if args.agent_name:
                self.agent_name = args.agent_name
            if args.save_storage:
                self.save_storage = args.save_storage
            if args.load_storage:
                self.load_storage = args.load_storage
            if args.storage_name:
                self.storage_name = args.storage_name

        self.experiment.set_glue_experiment(self)

        self.last_action = None

        if self.load_agent:
            prefix = self.agent_name
            self.agent = agent.load(prefix)
            try:
                print self.agent.storage.data
            except:
                pass
        elif self.load_storage:
            prefix = self.storage_name
            self.agent.init(None)
            self.agent.storage.load(prefix)

        try:
            self.experiment.run()
        finally:
            if self.save_agent:
                prefix = self.agent_name
                self.agent.save(prefix)
            elif self.save_storage:
                prefix = self.storage_name
                self.agent.storage.save(prefix)

    def init(self):
        self.episode_reward = 0
        self.agent.init(self.env.init())

    def start(self):
        observation = self.env.start()
        action = self.agent.start(observation)
        self.last_action = action
        self.episode_reward = 0
        return observation, action

    def step(self):
        rot = self.env.step(self.last_action)
        reward = rot["reward"]
        terminal = rot["terminal"]
        observation = rot.copy()
        del observation["reward"]
        del observation["terminal"]
        self.episode_reward += reward

        if terminal:
            self.agent.end(reward)
            return rot
        else:
            action = self.agent.step(reward, observation)
            self.last_action = action
        roat = rot.copy()
        roat["action"] = action
        return roat

    def episode(self):
        self.start()
        steps = 0
        terminal = False
        while not terminal:
            roat = self.step()
            terminal = roat["terminal"]
            steps += 1

    def return_reward(self):
        return self.episode_reward

    def agent_call(self, function_name, *args, **kwargs):
        return getattr(self.agent, "on_"+function_name)(*args, **kwargs)

    def agent_message(self, string):
        return self.agent.message(string)

    def env_call(self, function_name, *args, **kwargs):
        return getattr(self.env, "on_"+function_name)(*args, **kwargs)

    def env_message(self, string):
        return self.env.message(string)

    def cleanup(self):
        self.env.cleanup()
        self.agent.cleanup()

register_command(Run)

class RlGlueRun(Command):
    """Run the experiment using an rl glue server"""
    name = "rl_glue_run"

    def register(self, parser):
        parser.add_argument("--max-episodes", type=int, dest="max_episodes",
                            help="the maximum number of episodes to execute")
        parser.add_argument("--max-steps", type=int, dest="max_steps",
                            help="the maximum number of steps per "
                            "episodes to execute")


    def run(self, agent, env, experiment, args):
        self.agent = agent
        self.env = env
        self.experiment = experiment

        if args.max_episodes is not None:
            experiment.max_episodes = args.max_episodes
        if args.max_steps is not None:
            experiment.max_steps = args.max_steps

        agent_p = Process(target=start_agent,
                        args=(agent,))
        environment_p = Process(target=start_environment,
                              args=(env,))
        experiment_p = Process(target=start_experiment,
                             args=(experiment,))

        agent_p.start()
        environment_p.start()
        experiment_p.start()

        agent_p.join()
        environment_p.join()
        experiment_p.join()

if has_rlglue:
    register_command(RlGlueRun)

class RunAgent(Command):
    """Run the agent against an rl glue server"""
    name = "agent"

    def run(self, agent, env, experiment, args):
        start_agent(agent)
register_command(RunAgent)

class RunEnv(Command):
    """Run the environment against an rl glue server"""
    name = "environment"

    def run(self, agent, env, experiment, args):
        start_environment(env)
register_command(RunEnv)

class RunExperiment(Command):
    """Run the experiment against an rl glue server"""
    name = "experiment"

    def run(self, agent, env, experiment, args):
        start_experiment(experiment)
register_command(RunExperiment)

class Runner(object):
    def __init__(self, agent, env, experiment):
        self.agent = agent
        self.env = env
        self.experiment = experiment

    def run(self):
        parser = argparse.ArgumentParser(
            description='control and run RL experiments.')
        subparsers = parser.add_subparsers(dest="action")

        # here we can add generic options
        #parser.add_argument("--set", nargs=2, action="append",
        #                    help="set the value of a parameter")

        actions = {}
        for klass in _commands:
            klass_subparser = subparsers.add_parser(
                klass.name,
                help=klass.__doc__)
            a = klass()
            actions[a.name] = a
            a.register(klass_subparser)

        args = parser.parse_args()
        selected_action = actions[args.action]
        selected_action.run(self.agent, self.env, self.experiment, args)
