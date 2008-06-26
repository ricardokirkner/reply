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
    
class RL(object):
    def __init__(self, learner, storage, encoder, selector):
        self.learner = learner
        self.encoder = encoder
        self.selector = selector
        self.storage = storage
        storage.rl = learner.rl = encoder.rl = selector.rl = self
        self.total_steps = 0
        self.episodes = 0
        
        
    def new_episode(self, world, max_steps=1000):
        self.learner.new_episode()
        self.selector.new_episode()
        self.storage.new_episode()
        self.episodes += 1
        self.last_state = None
        self.current_state = world.get_initial_state()
        self.encoded_current_state = self.encoder.encode_state( self.current_state )
        self.total_reward = 0

    def step(self, world):

        if self.last_state:
            # get new state and perform learning
            next_state = world.get_state()
            encoded_next_state = self.encoder.encode_state( next_state )
            # observe the reward for this state
            reward = world.get_reward( next_state )
            self.total_reward += reward
            
            # perform the learning
            self.learner.update(
                self.encoded_current_state,
                self.encoded_action,
                reward,
                encoded_next_state,
                )
                
            
            self.current_state = next_state
            self.encoded_current_state = encoded_next_state
            
            self.total_steps += 1
            if world.is_final(self.current_state):
                return False

                    
        while True:
            try:
                # select an action using the current selection method
                self.encoded_action = self.selector.select_action(
                        self.encoded_current_state
                    )
                action = self.encoder.decode_action( self.encoded_action )
                
                # perform the action in the world
                world.do_action( self, action )
            except ActionNotPossible:
                continue
            break
        self.last_state = self.current_state
        return True
                
    def run(self, world, max_steps=1000):
        self.new_episode(world)
        self.step(world)
        for step in range(max_steps):
            if not self.step(world):
                break
            
        return self.total_reward, step
            
            

