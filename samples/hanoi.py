import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import reply
import math


DEBUG = False

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

    def fill(self, num_discs):
        self.discs = list(reversed(range(num_discs)))

    def empty(self):
        self.discs = []

    def pop(self):
        return self.discs.pop()

    def push(self, disc):
        self.discs.append(disc)

    def __repr__(self):
        return '<%s: number: %d discs: %s>' % (
            self.__class__.__name__, self.number, str(self.discs))

    def __contains__(self, disc):
        return disc in self.discs


class Hanoi(reply.World):
    def __init__(self, rl):
        super(Hanoi, self).__init__(rl)
        self.num_discs = getattr(rl, "num_discs", 5)
        self.num_pegs = getattr(rl, "num_pegs", 3)
        self.initial_peg = getattr(rl, "initial_peg", 0)
        self.pegs = [Peg(i) for i in xrange(self.num_pegs)]

    def is_final(self):
        is_final = False
        for peg in self.pegs:
            is_final |= self.discs_moved(peg)
        return is_final

    def do_action(self, action):
        from_peg_num, to_peg_num = action
        if DEBUG:
            print 'moving from peg %d to peg %d' % (from_peg_num, to_peg_num)
        from_peg = self.pegs[int(from_peg_num)]
        to_peg = self.pegs[int(to_peg_num)]

        if not from_peg.discs:
            if DEBUG:
                print "move invalid: empty"
            raise reply.ActionNotPossible()

        if to_peg.discs and from_peg.discs[-1] > to_peg.discs[-1]:
            if DEBUG:
                print "move invalid: too big"
            raise reply.ActionNotPossible()

        disc = from_peg.pop()
        to_peg.push(disc)
        if DEBUG:
            print "\t".join( [ str(peg.discs) for peg in self.pegs ] )

    def new_episode(self):
        # reset all pegs
        [peg.empty() for peg in self.pegs]
        # initialize
        self.pegs[self.initial_peg].fill(self.num_discs)

    def get_state(self):
        state = {}
        for disc in range(self.num_discs):
            state["disc_%d" % disc] = self.find_peg(disc)
        return state

    def discs_moved(self, peg):
        return peg.number != self.initial_peg and \
               peg.num_discs == self.num_discs

    def find_peg(self, disc):
        for i,peg in enumerate(self.pegs):
            if disc in peg:
                return i
        raise Exception("disc not found")

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, str(self.pegs))


class HanoiAgent(reply.Agent):
    num_pegs = 3
    num_discs = 5

    random_action_rate = 0

    world_class = Hanoi
    selector_class = reply.selector.EGreedySelector
    storage_class = reply.storage.TableStorage
    encoder_class = reply.encoder.DistanceEncoder

    def get_action_space(self):
        return [ reply.Dimension("from_peg", 0, self.num_pegs-1),
                 reply.Dimension("to_peg", 0, self.num_pegs-1) ]

    def get_state_space(self):
        return [ reply.Dimension("disc_%d" % disc, 0, self.num_pegs-1) for disc in xrange(self.num_discs) ]


class HanoiLearningAgent(HanoiAgent, reply.LearningAgent):
    learning_rate = 0.9
    learning_rate_decay = 0.999
    learning_rate_min = 0.0001
    value_discount = 0.8

    random_action_rate = 0.9
    random_action_rate_decay = 0.995

    learner_class = reply.learner.QLearner

    def get_reward(self):
        reward = 0
        for peg in self.world.pegs:
            if not peg.is_valid:
                # invalid disc layout
                if DEBUG:
                    print 'peg is invalid', peg.discs
                reward = -1
                break
            elif self.world.discs_moved(peg):
                # all discs moved to another peg
                if DEBUG:
                    print 'all discs moved to peg %d' % peg.number, peg.discs
                reward = 1
                break
        return reward


if __name__ == "__main__":
    agent = HanoiLearningAgent()
    experiment = reply.Experiment(agent)
    experiment.run()
