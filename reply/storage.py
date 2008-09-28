import cPickle as pickle
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
        if mappings is not None:
            self.state = numpy.zeros( (encoder.input_size, encoder.output_size) )
            for state, action in mappings.items():
                encoded_state = self.encoder.encode_state( state )
                self.state[encoded_state, action] = 1
        else:
            #self.state = numpy.random.random( (encoder.input_size, encoder.output_size) )
            self.state = numpy.zeros( (encoder.input_size, encoder.output_size) )
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


class DebugTableStorage(TableStorage):
    @property
    def median_hits(self):
        return numpy.median(self.debug_state)

    @property
    def count_hits(self):
        return numpy.sum(self.debug_state)

    def __init__(self, encoder, mappings=None):
        super(DebugTableStorage, self).__init__(encoder)
        self.debug_state = numpy.zeros( (encoder.input_size, encoder.output_size) )

    def store_value(self, state, action, new_value):
        self.value_change += abs(new_value-self.state[state, action])
        super(DebugTableStorage, self).store_value(state, action, new_value)
        self.sum_state_action_visits += self.debug_state[state, action]
        self.sum_state_visits += sum(self.debug_state[state])
        if sum(self.debug_state[state]) == 0:
            self.new_state_visits += 1
        if self.debug_state[state, action] == 0:
            self.new_state_action_visits += 1
        self.total_visits += 1
        self.debug_state[state, action]+=1

    def print_report(self):
        shape = self.debug_state.shape
        print "Dimension usage variation"
        print "Dimension usage median"
        print "Dimension usage average"
        print "Dimension coverage"
        print "State Coverage", len(numpy.nonzero(numpy.sum(self.debug_state,1)))/float(shape[0])
        #print self.debug_state
        print "State Coverage median", numpy.median(numpy.sum(self.debug_state, 1))
        sz = shape[0]*shape[1]
        print "State-Action Coverage", len(numpy.nonzero(self.debug_state.reshape(sz)))/float(sz)
        print "State-Action Coverage median", numpy.median(
                                                numpy.median(self.debug_state))
        print "new states visited:", self.new_state_visits, "%0.2f%%"%(100*self.new_state_visits/float(self.total_visits))
        print "new actions taken:", self.new_state_action_visits, "%0.2f%%"%(100*self.new_state_action_visits/float(self.total_visits))
        print "sum state visit", self.sum_state_visits, "%0.2f"%(self.sum_state_visits/self.total_visits)
        print "sum state action visit", self.sum_state_action_visits, "%0.2f"%(self.sum_state_action_visits/self.total_visits)
        print "value change", self.value_change, self.value_change/float(self.total_visits)

    def new_episode(self):
        super(DebugTableStorage, self).new_episode()
        self.new_state_visits = 0
        self.sum_state_visits = 0
        self.new_state_action_visits = 0
        self.sum_state_action_visits = 0
        self.total_visits = 0
        self.value_change = 0

