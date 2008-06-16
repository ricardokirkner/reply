import cPickle as pickle
import numpy
import random

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


class Encoder(object):
    def __init__(self, problem):
        """
        Problem must implement 'get_problem_space' and 'get_action_space'
        problem must not be stored as an instance variable (pickle problems
        may arise)
        """
        self.problem_space = problem.get_problem_space()
        self.action_space = problem.get_action_space()
        
    def encode_state(self, state):
        """
        The parameters are received in world-encoding and returned
        in rl-encoding
        """    
        raise NotImplementedError()
        
    def decode_action(self, action_n):
        """
        The parameters are received in rl-encoding and returned
        in world-encoding
        """    
        raise NotImplementedError()
        
class DistanceEncoder(Encoder):
    def get_space_size(self, space):
        pz = 1
        for dim in space:
            pz *= len(dim)
        return pz
        
    def __init__(self, problem):
        super(DistanceEncoder, self).__init__(problem)
        self.input_size = self.get_space_size(self.problem_space)
        self.output_size = self.get_space_size(self.action_space)
        
    def encode_state(self, state):
        """
        do base convertion from variable-base state to 10-based state number
        """
        m = 1
        state_n = 0
        for dim, v in zip(self.problem_space, state):
            n = numpy.argmin( (dim-v)**2 )
            state_n += n*m
            m *= len(dim)
        return state_n
        
    def decode_action(self, action_n):
        """
        do base conversion from 10-base action number to variable-base
        action
        """
        m = 1

        action = numpy.zeros( len(self.action_space) )
        for i, dim in enumerate(self.action_space):
            n = len(dim)
            j = ( action_n/m ) % n
            m *= n
            action[i] = dim[j]
            #print n, j, m, dim[j], dim
        return action


class Storage(object):  
    def __init__(self, encoder):
        """
        Encoder provides the size of the problem
        """
        self.encoder = encoder
        
    def new_episode(self):
        """
        This function is called whenever a new episode is started.
        Histories and traces should be cleared here. 
        """
        pass
        
    def store_value(self, state, action, new_value):
        """
        The parameters are received in rl-encoding
        """
        raise NotImplementedError()
    
    def get_value(self, state, action):
        """
        The parameters are received in rl-encoding
        """
        raise NotImplementedError()
        
    def get_state_values(self, state):
        """
        Returns an array of the action values for this state
        The parameters are received in rl-encoding
        """
        raise NotImplementedError()
        
    def get_max_value(self, state):
        """
        The parameters are received in rl-encoding
        """
        raise NotImplementedError()

    def load(self, filename):
        """
        This function retrieves a persisted storage from a file.
        """
        raise NotImplementedError()

    def dump(self, filename):
        """
        This function persists the storage to a file.
        """
        raise NotImplementedError()

        
class TableStorage(Storage):
    def __init__(self, encoder):
        self.state = numpy.zeros( (encoder.input_size, encoder.output_size) )
        #self.state = numpy.random.random( (encoder.input_size, encoder.output_size) )
        
    def store_value(self, state, action, new_value):
        self.state[state, action] = new_value
        
    def get_value(self, state, action):
        return self.state[state, action]
        
    def get_max_value(self, state):
        return max(self.state[ state ])
        
    def get_state_values(self, state):
        return self.state[ state ]

    def load(self, filename):
        file = open(filename, 'rb')
        self.state = pickle.load(file)

    def dump(self, filename):
        file = open(filename, 'wb')
        pickle.dump(self.state, file)

        
class Learner(object):
    def __init__(self, storage):
        self.storage = storage
        
    def new_episode(self):
        """
        This function is called whenever a new episode is started.
        Histories and traces should be cleared here. This should
        also propagate the event to the storage.
        """
        self.storage.new_episode()
        
    def update(self, state, action, reward, next_state):
        """
        The parameters are received in rl-encoding
        """
        raise NotImplementedError()
        
class QLearner(Learner):
    def __init__(self, storage, alpha, gamma, alpha_decay = 1, min_alpha=None):
        """
        implements Q-Learning
        alpha: learning rate
        gamma: value discount
        """
        self.storage = storage
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.min_alpha = min_alpha
        self.gamma = gamma
        
    def new_episode(self):
        super(QLearner, self).new_episode()
        self.alpha *= self.alpha_decay
        if self.min_alpha is not None:
            self.alpha = max(self.min_alpha, self.alpha)
        
    def update(self, state, action, reward, next_state):
        prev_value = self.storage.get_value(state, action)
        max_value_next = self.storage.get_max_value( next_state )
        
        new_value = (
            prev_value + self.alpha *  
            ( reward + self.gamma*max_value_next - prev_value )       
            )
            
        #print prev_value, "->", new_value, 
        #print "(r=%i, a=%i)"%(reward, action)
        #print "max_next", max_value_next
        self.storage.store_value(state, action, new_value)
            
class Selector(object):
    def new_episode(self):
        """
        This function is called whenever a new episode is started.
        Histories and traces should be cleared here.
        """
        pass
        
    def select_action(self, action_value_array):
        """
        Implements an action selection procedure. The input parameter is
        an array of the values corresponding to the actions.
        The parameters are received in rl-encoding
        """
        raise NotImplementedError()
        
class EGreedySelector(Selector):
    def __init__(self, epsilon, decay=1):
        self.epsilon = epsilon
        self.decay = decay
        
    def new_episode(self):
        self.epsilon *= self.decay
        
    def select_action(self, action_value_array):
        if random.random() < self.epsilon:
            #print "R",
            action = random.randint(0, numpy.size(action_value_array)-1)
        else:
            #print action_value_array,
            action = numpy.argmax( action_value_array )
        #print action
        return action
        
class World(object):
    def get_action_space(self):
        """
        Returns a list of dimensions where each one has the list of posible
        values for that dimension. A world encoded representation of an action
        is a list with one value picked from each dimension.
        """
        raise NotImplementedError()

    def get_problem_space(self):
        """
        Returns a list of dimensions where each one has the list of posible
        values for that dimension. A world encoded representation of a state
        is a list with one value picked from each dimension.
        """
        raise NotImplementedError()
        
    def get_initial_state(self):
        """
        Initializes the world and returns the initial state
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
    def __init__(self, learner, encoder, selector):
        self.learner = learner
        self.encoder = encoder
        self.selector = selector
        self.total_steps = 0
        self.episodes = 0
        self.last_state = None
        
    def new_episode(self, world, max_steps=1000):
        self.learner.new_episode()
        self.selector.new_episode()
        self.episodes += 1
        
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
                encoded_action,
                reward,
                encoded_next_state
                )
                
            
            self.current_state = next_state
            self.encoded_current_state = encoded_next_state
            
            self.total_steps += 1



        value_array = self.learner.storage.get_state_values( self.current_state )
                    
        while True:
            try:
                # select an action using the current selection method
                encoded_action = self.selector.select_action(
                        value_array
                    )
                action = self.encoder.decode_action( encoded_action )
                
                # perform the action in the world
                world.do_action( self, action )
            except ActionNotPossible:
                continue
            break
        self.last_state = self.current_state
                
    def run(self, world, max_steps=1000):

        self.new_episode()
        for step in range(max_steps):
            self.step(world)
            if world.is_final(self.current_state):
                break
        return self.total_reward, step
            
            

