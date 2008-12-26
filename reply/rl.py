"""Agent and World classes."""
import numpy

from contrib import argparse

__all__ = ["Dimension", "World", "ActionNotPossible", "Episode",
           "Agent", "Experiment"]

class Dimension(object):

    """Dimension base class."""

    def __init__(self, name, start, end, points=None):
        """Initialize the dimension.

        Defines a dimension for a problem and a discretization for it.

        Arguments:
        start -- minimun value included in range
        end   -- maximum value included in range

        The range will have 'points' points.

        """
        self.name = name
        if points is None:
            points = abs(end-start)+1
        if points == 1:
            self.points = numpy.array([0])
        else:
            end = float(end)
            start = float(start)
            step_size = (end - start)/(points-1)
            d = numpy.zeros(points)
            for i in range(points):
                d[i] = start+step_size * i
            self.points = d

    def __len__(self):
        """Return the number of points in the dimension."""
        return len(self.points)

    def __iter__(self):
        """Return an iterator over the dimension."""
        return self.points.__iter__()


class World(object):

    """World base class."""

    def __init__(self, rl):
        """Initialize the world.

        Arguments:
        rl -- an instance of the agent

        """
        self.rl = rl

    def new_episode(self):
        """Start a new episode."""
        raise NotImplementedError()

    def end_episode(self):
        """End the current episode."""
        pass

    def is_final(self):
        """Return True if the current state is a final state."""
        return False

    def get_state(self):
        """Return the current state of the world."""
        raise NotImplementedError()

    def do_action(self, solver, action):
        """Perform an action in the world.

        Arguments:
        solver -- instance of RL performing the learning
        action -- action to perform. Received in world encoding.
        """
        raise NotImplementedError()


class ActionNotPossible(Exception):

    """Exception representing an impossible action."""

    pass


class Episode(object):

    """Episode base class."""

    def __str__(self):
        """Return a string representation of the episode."""
        return ", ".join("%s:%s"%(k, v) for k, v in self.__dict__.items())


class Agent(object):
    """Agent class.

    An agent must declare the following attributes:

    world_class    -- the class representing the world the agent lives in
    selector_class -- the class used for selecting actions
    encoder_class  -- the class used for converting between rl- and world encodings
    storage_class  -- the class used for storing the agent's data
    learner_class  -- the class used for the learning part of the agent

    """
    world_class = None
    selector_class = None
    encoder_class = None
    storage_class = None
    learner_class = None

    max_steps = 10000

    def __init__(self):
        """Initialize the agent."""
        self.world = self.world_class(self)
        self.selector = self.selector_class(self)
        self.encoder = self.encoder_class(self)
        self.storage = self.storage_class(self)
        self.learner = self.learner_class(self)
        self.episodes = 0

    def new_episode(self):
        """Start a new episode."""
        self.current_episode = Episode()
        self.episodes += 1
        self.current_episode.episode = self.episodes

        self.current_episode.steps = 0
        self.selector.new_episode()
        self.storage.new_episode()
        self.encoder.new_episode()
        self.world.new_episode()
        self.learner.new_episode()
        self.current_episode.total_reward = 0

        self.last_state = None
        self.current_state = self.encoder.encode_state(self.world.get_state())

    def end_episode(self):
        """End the current episode."""
        self.selector.end_episode()
        self.storage.end_episode()
        self.encoder.end_episode()
        self.world.end_episode()

    def step(self):
        """Perform a step in the world."""
        world = self.world
        if self.last_state is not None:
            # get new state
            next_state = self.encoder.encode_state(world.get_state())

            reward = self.get_reward()
            self.current_episode.total_reward += reward

            # perform the learning
            self.learner.update(
                self.current_state,
                self.current_action,
                reward,
                next_state,
                )

            # update current state
            self.current_state = next_state

            # update step count
            self.current_episode.steps += 1

            # test episode halting condition
            if world.is_final():
                self.end_episode()
                return False

        # select next action to execute
        while True:
            try:
                # select an action using the current policy
                self.current_action = self.selector.select_action(
                    self.current_state)

                # perform the action in the world
                world.do_action(self.encoder.decode_action(self.current_action))
            except ActionNotPossible:
                continue
            break

        # remember state for next step
        self.last_state = self.current_state
        return True

    def run_episode(self):
        """Perform a full episode in the world.

        Arguments:
        max_steps -- maximum number of steps to perform

        """
        self.new_episode()
        step = 0
        while step < self.max_steps or self.max_steps < 0:
            if not self.step():
                break
            step += 1

        return self.current_episode

    def _step_hook(self, next_state):
        """Hook into the step.

        Arguments:
        next_state -- state achieved after performing the action

        """
        pass

    def get_reward(self):
        """Return the reward obtainer after performing an action."""
        raise NotImplementedError()




class Command(object):
    #: the command name
    name = "replace_me"

    def register(self, parser):
        """Register this command in the parser subparser"""
        raise NotImplementedError()

    def run(self, options):
        """run the command with options"""
        raise NotImplementedError()

_commands = []
def register_command(command_klass):
    _commands.append(command_klass)

class Run(Command):
    """Run the experiment in foreground"""
    name = "run"

    def register(self, parser):
        parser.add_argument("--max-episodes", type=int, dest="max_episodes",
                            help="the maximum number of episodes to execute")


    def run(self, agent, args):
        agent.new_episode()
        while True:
            episode = agent.run_episode()
            print episode
            if args.max_episodes is not None:
                if episode.episode >= args.max_episodes:
                    break

register_command(Run)


class Experiment(object):
    """Experiment base class."""

    def __init__(self, agent):
        """Initialize the experiment.

        Arguments:
        agent -- an instance of the agent

        """
        self.agent = agent

    def run(self):
        parser = argparse.ArgumentParser(
            description='control and run RL experiments.')
        subparsers = parser.add_subparsers(dest="action")

        parser.add_argument("--set", nargs=2, action="append",
                            help="set the value of a parameter")

        actions = {}
        for klass in _commands:
            klass_subparser = subparsers.add_parser(
                klass.name,
                help=klass.__doc__)
            a = klass()
            actions[a.name] = a
            a.register(klass_subparser)

        args = parser.parse_args()
        if args.set is not None:
            for name, value in args.set:
                setattr(self.agent, name, eval(value))
        selected_action = actions[args.action]
        selected_action.run(self.agent, args)
