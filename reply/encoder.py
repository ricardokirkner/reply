"""Encoder classes."""
import numpy

class Encoder(object):

    """Encoder base class."""

    def __init__(self, state_space, action_space):
        """Initialize the encoder.

        state_space is a list of dimensions where each one has the list of posible
        values for that dimension. A world encoded representation of a state
        is a list with one value picked from each dimension.

        action_space is a list of dimensions where each one has the list of posible
        values for that dimension. A world encoded representation of an action
        is a list with one value picked from each dimension.

        """
        self.state_space = state_space
        self.action_space = action_space
        
    def encode_state(self, state):
        """Return the rl-encoding for a given world-encoded state."""    
        raise NotImplementedError()
        
    def decode_action(self, action_n):
        """Return the world-encoding for a given rl-encoded action."""    
        raise NotImplementedError()
        

class DistanceEncoder(Encoder):

    """Encoder that does base conversion from variable-base to 10-base."""

    def __init__(self, state_space, action_space):
        super(DistanceEncoder, self).__init__(state_space, action_space)
        self.input_size = self.get_space_size(self.state_space)
        self.output_size = self.get_space_size(self.action_space)
        
    def encode_state(self, state):
        """Return the rl-encoding for a given world-encoded state.

        Do base convertion from variable-base state to 10-based state number.

        """
        m = 1
        encoded_state = 0
        for dim, v in zip(self.state_space, state):
            n = numpy.argmin( (dim-v)**2 )
            encoded_state += n*m
            m *= len(dim)
        return encoded_state
        
    def decode_action(self, encoded_action):
        """Return the world-encoding for a given rl-encoded action.

        Do base conversion from 10-base action number to variable-base action.

        """
        m = 1

        action = numpy.zeros( len(self.action_space) )
        for i, dim in enumerate(self.action_space):
            n = len(dim)
            j = ( encoded_action/m ) % n
            m *= n
            action[i] = dim[j]
            #print n, j, m, dim[j], dim
        return action

    def get_space_size(self, space):
        """Return the size of the given space."""
        size = 1
        for dim in space:
            size *= len(dim)
        return size
        

if __name__ == "__main__":
    import rl
    encoder = DistanceEncoder( 
        [ rl.dimension(0,3,4) ],
        [ rl.dimension(0,3,4) ] 
        )
    for i in range(4):
        print i, "-->", encoder.encode_state( [i] )
        
    print "--"*20
    encoder = DistanceEncoder( 
        [ rl.dimension(0,3,4), rl.dimension(0,5,6) ],
        [ rl.dimension(0,3,4) ] 
        )
    for i in range(4):
        for j in range(6):
            print i, j, "-->", encoder.encode_state( (i, j) )
