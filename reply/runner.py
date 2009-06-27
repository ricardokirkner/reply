from multiprocessing import Process

from reply.contrib import argparse
from reply.glue import start_agent
from reply.glue import start_environment
from reply.glue import start_experiment

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

class Run(Command):
    """Run the experiment in foreground"""
    name = "run"

    def register(self, parser):
        parser.add_argument("--max-episodes", type=int, dest="max_episodes",
                            help="the maximum number of episodes to execute")
        parser.add_argument("--max-steps", type=int, dest="max_steps",
                            help="the maximum number of steps per "
                            "episodes to execute")


    def run(self, agent, env, experiment, args=None):
        self.agent = agent
        self.env = env
        self.experiment = experiment

        if args is not None:
            if args.max_episodes is not None:
                experiment.max_episodes = args.max_episodes
            if args.max_steps is not None:
                experiment.max_steps = args.max_steps

        self.experiment.set_glue_experiment(self)

        self.last_action = None
        self.experiment.run()

    def init(self):
        self.episode_reward = 0
        return self.agent.init(self.env.init())

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
