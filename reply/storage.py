import numpy

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
    def __init__(self, encoder, mappings=None):
        """parameters:
        @mappings: a dictionary with two keys, True and False, that contain a set of (state, action) pairs
        """
        super(TableStorage, self).__init__(encoder)
        self.state = numpy.zeros( (encoder.input_size, encoder.output_size) )
        if mappings is not None:
            for state, action in mappings.items():
                encoded_state = self.encoder.encode_state( state )
                self.state[encoded_state, action] = 1
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

        