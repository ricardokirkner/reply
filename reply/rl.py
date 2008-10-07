import cPickle as pickle
import numpy
import random

__all__ = ["dimension", "World", "ActionNotPossible", "Agent", "LearningAgent"]

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
        self.storage = storage
        self.encoder = encoder
        self._selector = selector
        # bind storage to selector
        self._selector.storage = storage

    @apply
    def selector():
        def fget(self):
            if self._selector is not None:
                return self._selector
            else:
                raise ValueError('Selector has not yet been set.')

        def fset(self, value):
            self._selector = value
            # bind storage to selector
            self._selector.storage = self.storage

        return property(**locals())

    def get_action(self, state):
        encoded_state = self.encoder.encode_state(state)
        encoded_action = self.selector.select_action(encoded_state)
        action = self.encoder.decode_action(encoded_action)
        return action

    def get_value(self, state, action):
        encoded_state = self.encoder.encode_state(state)
        encoded_action = self.encoder.encode_action(action)
        return self.storage.get_value(encoded_state, encoded_action)

    def get_max_value(self, state):
        encoded_state = self.encoder.encode_state(state)
        return self.storage.get_max_value(encoded_state)

    def new_episode(self):
        self.selector.new_episode()
        self.storage.new_episode()

    def update(self, state, action, value):
        encoded_state = self.encoder.encode_state(state)
        encoded_action = self.encoder.encode_action(action)
        self.storage.store_value(encoded_state, encoded_action, value)

    def load(self, filename):
        self.storage.load(filename)

    def dump(self, filename):
        self.storage.dump(filename)


class Agent(object):

    def __init__(self, policy, world=None):
        self.policy = policy
        self._world = world
        self.total_steps = 0
        self.episodes = 0

    @apply
    def world():
        def fget(self):
            if self._world is not None:
                return self._world
            else:
                raise ValueError('World has not yet been set.')

        def fset(self, value):
            self._world = value

        return property(**locals())

    def new_episode(self):
        self.policy.new_episode()
        self.episodes += 1
        self.last_state = None
        self.current_state = self.world.get_initial_state()

    def step(self):
        world = self.world
        if self.last_state:
            # get new state 
            next_state = world.get_state()

            # hook into the step 
            self._step_hook(next_state)

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

    def run(self, max_steps=1000):
        self.new_episode()
        self.step()
        for step in range(max_steps):
            if not self.step():
                break

        return step

    def _step_hook(self, next_state):
        pass


class LearningAgent(Agent):

    def __init__(self, policy, learner):
        super(LearningAgent, self).__init__(policy)
        self._learner = learner
        # bind policy to learner
        self._learner.policy = policy

    @apply
    def learner():
        def fget(self):
            if self._learner is not None:
                return self._learner
            else:
                raise ValueError('Learner has not yet been set.')

        def fset(self, value):
            self._learner = value
            # bind policy to learner
            self._learner.policy = self.policy

        return property(**locals())

    def new_episode(self):
        self.learner.new_episode()
        self.total_reward = 0
        super(LearningAgent, self).new_episode()

    def _step_hook(self, next_state):
        # observe the reward for this state
        reward = self.world.get_reward(next_state)
        self.total_reward += reward

        # perform the learning
        self.learner.update(
            self.current_state,
            self.current_action,
            reward,
            next_state,
            )

    def run(self, max_steps=1000):
        steps = super(LearningAgent, self).run(max_steps=max_steps)
        return self.total_reward, steps

