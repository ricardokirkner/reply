import sys
sys.path.insert(0, '..')

from reply import *
import math

NUM_PEGS = 3
NUM_DISCS = 3

def prime_generator():
    # yield the only even prime
    yield 2
    # yield odd primes
    current = 3
    while True:
        if is_odd_prime(current):
            yield current
        current += 2

def is_odd_prime(number):
    is_prime = True
    sieve = int(math.sqrt(number))
    for i in xrange(2, sieve+1):
        if number % i == 0:
            is_prime = False
            break
    return is_prime

def max_peg_state(num_discs):
    peg = Peg(0)
    for disc in xrange(1,num_discs+1):
        peg.discs.append(disc)
    return peg.state


class Peg(object):
    def __init__(self, number):
        self.number = number
        self.discs = []
    
    @property
    def is_valid(self):
        return sorted(self.discs, reverse=True) == self.discs

    @property
    def num_discs(self):
        return len(self.discs)

    @property
    def state(self):
        state = 1
        primes = prime_generator()
        for disc in self.discs:
            prime = primes.next()
            state *= prime**disc
        return state


    def fill(self, num_discs):
        self.discs = range(num_discs, 0, -1)

    def empty(self):
        self.discs = []

    def pop(self):
        return self.discs.pop()

    def push(self, disc):
        self.discs.append(disc)

    def __repr__(self):
        return '<%s: number: %d discs: %s>' % (self.__class__.__name__, self.number, str(self.discs))

import operator
def factorial(n):
    return reduce(operator.mul, xrange(1,n+1), 1)

def permutations(n, r):
    return float(factorial(n)) / factorial(n-r)


class Player(rl.World):
    def __init__(self, num_discs, num_pegs=NUM_PEGS, initial_peg=0):
        self.num_discs = num_discs
        self.num_pegs = num_pegs
        self.initial_peg = initial_peg
        self.pegs = [Peg(i) for i in xrange(num_pegs)]
        # initialize
        self.pegs[self.initial_peg].fill(self.num_discs)

    def get_action_space(self):
        return [ rl.dimension(0, 2, 3), rl.dimension(0, 2, 3) ]
    
    def get_problem_space(self):
#        max_value = max_peg_state(self.num_discs)
#        #num_values = sum([permutations(self.num_discs, discs) for discs in xrange(self.num_discs+1)])
#        num_values = 2**(self.num_discs+1)
#        return [ rl.dimension(1, max_value, num_values),
#                 rl.dimension(1, max_value, num_values),
#                 rl.dimension(1, max_value, num_values) ]
        return [ rl.dimension(0, self.num_pegs, self.num_pegs+1) for i in xrange(self.num_discs) ] 
        
    def get_reward(self, state):
        reward = 0
        for peg in self.pegs:
            if not peg.is_valid:
                # invalid disc layout
                print 'peg is invalid', peg.discs
                reward = -1
                break
            elif self.discs_moved(peg):
                # all discs moved to another peg
                #print 'all discs moved to peg %d' % peg.number, peg.discs
                reward = 1
                break
        return reward
        
    def discs_moved(self, peg):
        return peg.number != self.initial_peg and \
               peg.num_discs == self.num_discs

    def is_final(self, state):
        is_final = False
        for peg in self.pegs:
            is_final |= not peg.is_valid or self.discs_moved(peg)
        return is_final
        
    def do_action(self, solver, action):
        #print 'action', action
        from_peg_num, to_peg_num = action
        print 'moving from peg %d to peg %d' % (from_peg_num, to_peg_num)
        from_peg = self.pegs[int(from_peg_num)]
        to_peg = self.pegs[int(to_peg_num)]
        
        if len(from_peg.discs) < 1:
            raise rl.ActionNotPossible()
        if len(to_peg.discs) > 0 and \
           from_peg.discs[-1] > to_peg.discs[-1]:
            raise rl.ActionNotPossible()
        disc = from_peg.pop()
        to_peg.push(disc)
        return self.pegs
        
    def get_initial_state(self):
        return self.get_state()

    def get_state(self):
        return [peg.state for peg in self.pegs]

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, str(self.pegs))



if __name__ == "__main__":
    p = Player(NUM_DISCS)
    e = encoder.DistanceEncoder(
            p.get_problem_space(),
            p.get_action_space(),
        )
    
    import sys
    if len(sys.argv) < 2:
        print "Usage: %s [learn|eval]" % __file__
        sys.exit(1)
    else:
        action = sys.argv[1]
        try:
            num_steps = int(sys.argv[2])
        except:
            num_steps = 1000
        FILENAME = 'hanoi.exp'

    if action == 'learn':
        r = rl.RL(
                learner.QLearner(0.9, 0.8, 0.999, 0.0001 ),
                storage.TableStorage(e),
                e, 
                selector.EGreedySelector(0.5, 1)
            )
        try:
            r.storage.load(FILENAME)
        except:
            # file did not exist
            pass
    elif action == 'eval':
        r = rl.RL(
                learner.QLearner(1.0, 0.0),
                storage.TableStorage(e),
                e,
                selector.EGreedySelector(0)
            )
        try:
            r.storage.load(FILENAME)
        except:
            # file did not exist
            print 'saved state was not found. evaluating does not make sense'
            sys.exit(1)
    else:
        print "Usage: %s [learn|eval]" % __file__
        sys.exit(1)
        
    #initial = r.storage.state

    for episode in range(num_steps):    
        p = Player(NUM_DISCS)
        #print 'Initial state:', p
        total_reward,steps = r.run(p)
        #print 'Final state:', p
    
        print 'Espisode:',episode,'  Steps:',steps,'  Reward:',total_reward,' epsilon:',r.selector.epsilon, "alpha:", r.learner.alpha
        #print '-'*80
    #final = r.storage.state
    #print 'learned?', initial != final
    if action == 'learn':
        r.storage.dump(FILENAME)
        
