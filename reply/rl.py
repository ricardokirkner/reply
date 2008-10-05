import cPickle as pickle
import numpy
import random

__all__ = ["dimension", "World", "ActionNotPossible", "RL"]

def dimension(start, end, points):
    """
    Defines a dimension for a problem and a discretization for it.
    Start and end are the minimum and maximum values and will be included
    in the range. The range will have 'points' points.
    """
    if points == 1:
        return numpy.array([0])
    end = float(end)
    start = float(start)
    step_size = (end - start)/(points-1)
    d = numpy.zeros(points)
    for i in range(points):
        d[i] = start+step_size * i
    return d





class World(object):
    def get_initial_state(self):
        """
        Initializes the world and returns the initial state
        """
        raise NotImplementedError()

    def get_state(self):
        """
        Returns current state
        """
        raise NotImplementedError()


    def do_action(self, solver, action):
        """
        Performs action in the world.
        solver is the instance of RL performing the learning.
        The action parameter is received in world encoding.
        """
        raise NotImplementedError()

    def get_reward(self):
        raise NotImplementedError()


class ActionNotPossible(Exception):
    pass


class Policy(object):

    def __init__(self, storage, encoder, selector):
        self.encoder = encoder
        self.selector = selector
        self.storage = storage

    @apply
    def agent():
        def fget(self):
            return self.encoder.agent

        def fset(self, value):
            self.encoder.agent = value
            self.selector.agent = value
            self.storage.agent = value

        return property(**locals())

    def get_action(self, state):
        encoded_state = self.encoder.encode_state(state)
        encoded_action = self.selector.select_action(encoded_state)
        action = self.encoder.decode_action(encoded_action)
        return action

    def new_episode(self):
        self.selector.new_episode()
        self.storage.new_episode()

    def get_value(self, state, action):
        encoded_state = self.encoder.encode_state(state)
        encoded_action = self.encoder.encode_action(action)
        return self.storage.get_value(encoded_state, encoded_action)

    def get_max_value(self, state):
        encoded_state = self.encoder.encode_state(state)
        return self.storage.get_max_value(encoded_state)

    def update(self, state, action, value):
        encoded_state = self.encoder.encode_state(state)
        encoded_action = self.encoder.encode_action(action)
        self.storage.store_value(encoded_state, encoded_action, value)

    def load(self, filename):
        self.storage.load(filename)

    def dump(self, filename):
        self.storage.dump(filename)


class Agent(object):

    def __init__(self, policy):
        self.policy = policy
        self.policy.agent = self
        self.total_steps = 0
        self.episodes = 0

    def new_episode(self, world):
        self.policy.new_episode()
        self.episodes += 1
        self.last_state = None
        self.current_state = world.get_initial_state()

    def step(self, world):
        if self.last_state:
            # get new state 
            next_state = world.get_state()

            # hook into the step 
            self._step_hook(world, next_state)

            # update current state
            self.current_state = next_state

            # update step count
            self.total_steps += 1

            # test episode halting condition
            if world.is_final(self.current_state):
                return False

        # select next action to execute
        while True:
            try:
                # select an action using the current policy
                self.current_action = self.policy.get_action(self.current_state)

                # perform the action in the world
                world.do_action(self, self.current_action)
            except ActionNotPossible:
                continue
            break

        # remember state for next step
        self.last_state = self.current_state
        return True

    def run(self, world, max_steps=1000):
        self.new_episode(world)
        self.step(world)
        for step in range(max_steps):
            if not self.step(world):
                break

        return step

    def _step_hook(self, world, next_state):
        pass


class LearningAgent(Agent):

    def __init__(self, learner, policy):
        super(LearningAgent, self).__init__(policy)
        self.learner = learner
        learner.agent = self

    def new_episode(self, world):
        self.learner.new_episode()
        self.total_reward = 0
        super(LearningAgent, self).new_episode(world)

    def _step_hook(self, world, next_state):
        # observe the reward for this state
        reward = world.get_reward(next_state)
        self.total_reward += reward

        # perform the learning
        self.learner.update(
            self.policy,
            self.current_state,
            self.current_action,
            reward,
            next_state,
            )

    def run(self, world, max_steps=1000):
        steps = super(LearningAgent, self).run(world, max_steps=max_steps)
        return self.total_reward, steps

# backwards compatibility
RL = LearningAgent
