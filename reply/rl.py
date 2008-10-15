import cPickle as pickle
import numpy
import random

__all__ = ["Dimension", "World", "ActionNotPossible", "Episode",
           "Agent", "LearningAgent", "Experiment"]

class Dimension(object):
    def __init__(self, name, start, end, points=None):
        """
        Defines a dimension for a problem and a discretization for it.
        Start and end are the minimum and maximum values and will be included
        in the range. The range will have 'points' points.
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
        return len(self.points)

class World(object):
    def __init__(self, rl):
        self.rl = rl
        
    def new_episode(self):
        """
        Initializes the world and returns the initial state
        """
        raise NotImplementedError()
    
    def is_final(self):
        return False

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

class ActionNotPossible(Exception):
    pass

class Episode(object):
    def __str__(self):
        return ", ".join("%s:%s"%(k,v) for k,v in self.__dict__.items())

class Agent(object):
    def __init__(self):
        self.world = self.world_class(self)
        self.selector = self.selector_class(self)
        self.encoder = self.encoder_class(self)
        self.storage = self.storage_class(self)
        self.episodes = 0

    def new_episode(self):
        self.current_episode = Episode()
        self.episodes += 1
        self.current_episode.episode = self.episodes

        self.current_episode.steps = 0
        self.selector.new_episode()
        self.storage.new_episode()
        self.encoder.new_episode()
        self.world.new_episode()
        
        self.last_state = None
        self.current_state = self.encoder.encode_state(self.world.get_state())

    def end_episode(self):
        self.selector.end_episode()
        self.storage.end_episode()
        self.encoder.end_episode()
        self.world.end_episode()
        
        
    def step(self):
        world = self.world
        if self.last_state is not None:
            # get new state 
            next_state = self.encoder.encode_state(world.get_state())

            # hook into the step 
            self._step_hook(next_state)

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
                self.current_action = self.selector.select_action(self.current_state)

                # perform the action in the world
                world.do_action(self, self.encoder.decode_action(self.current_action))
            except ActionNotPossible:
                continue
            break

        # remember state for next step
        self.last_state = self.current_state
        return True

    def run(self, max_steps=1000):
        self.new_episode()
        step = 0
        while step < max_steps or max_steps < 0:
            if not self.step():
                break
            step += 1
        
        return self.current_episode

    def _step_hook(self, next_state):
        pass
    


class LearningAgent(Agent):
    def __init__(self):
        super(LearningAgent, self).__init__()
        self.learner = self.learner_class(self)

    def new_episode(self):
        super(LearningAgent, self).new_episode()
        self.learner.new_episode()
        self.current_episode.total_reward = 0

    def _step_hook(self, next_state):
        super(LearningAgent, self)._step_hook(next_state)
        # observe the reward for this state
        reward = self.get_reward()
        self.current_episode.total_reward += reward

        # perform the learning
        self.learner.update(
            self.current_state,
            self.current_action,
            reward,
            next_state,
            )
        
    def get_reward(self):
        raise NotImplementedError()


class Experiment(object):
    def __init__(self, agent):
        self.agent = agent
        
    def run(self):
        self.agent.new_episode()
        step = 0
        while True:
            episode = self.agent.run()
            print episode

