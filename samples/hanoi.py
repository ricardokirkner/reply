import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import itertools
import math

from reply.agent import LearningAgent
from reply.datatypes import Integer, Model, Space
from reply.encoder import SpaceEncoder
from reply.environment import Environment
from reply.experiment import Experiment
from reply.learner import QLearner
from reply.policy import EGreedyPolicy
from reply.storage import TableStorage

DEBUG, INFO, NONE = range(3)
LOG_LEVEL = NONE


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


num_pegs = 3
num_discs = 5

discs = ['disc_%d' % disc for disc in xrange(num_discs)]
spec = {}
for disc in discs:
    spec[disc] = Integer(0, num_pegs-1)
observations = Space(spec)
actions = Space({'from_peg': Integer(0, num_pegs-1),
                 'to_peg': Integer(0, num_pegs-1)})
hanoi_model = Model(observations, actions)


class HanoiSpaceEncoder(SpaceEncoder):
    def encode(self, item):
        item = super(HanoiSpaceEncoder, self).encode(item)

        values = []
        for parameter in self.space.spec.values():
            values.append(range(parameter.size))
        items = list(itertools.product(*values))
        try:
            idx = items.index(item)
            encoded_item = idx
        except ValueError:
            raise ValueError("invalid encoded item: %s" % str(item))

        return (encoded_item,)

    def decode(self, encoded_item):
        values = []
        for parameter in self.space.spec.values():
            values.append(range(parameter.size))
        items = list(itertools.product(*values))
        try:
            idx = encoded_item[0]
            item = items[idx]
        except ValueError:
            raise ValueError("invalid encoded item: %s" % str(encoded_item))

        item = super(HanoiSpaceEncoder, self).decode(item)
        return item


class HanoiActionEncoder(SpaceEncoder):
    pass


class HanoiAgent(LearningAgent):
    model = hanoi_model
    state_encoder_class = HanoiSpaceEncoder
    action_encoder_class = HanoiSpaceEncoder
    storage_class = TableStorage
    policy_class = EGreedyPolicy
    learner_class = QLearner

    learning_rate = 0.9
    learning_rate_decay = 0.999
    learning_rate_min = 0.0001
    value_discount = 0.8
    random_action_rate = 0.9
    random_action_rate_decay = 0.995

    def cleanup(self):
        import pprint
        pprint.pprint(self.learner.policy.get_mappings())


class HanoiEnvironment(Environment):
    problem_type = "episodic"
    discount_factor = 0.8
    rewards = Integer(-1, 1)
    model = hanoi_model

    def init(self):
        self.num_discs = num_discs # FIXME: make it a variable
        self.num_pegs = num_pegs # FIXME: make it a variable
        self.initial_peg = 0 # FIXME: make it a variable 
        self.pegs = [Peg(i) for i in xrange(self.num_pegs)]

    def start(self):
        # reset all pegs
        [peg.empty() for peg in self.pegs]
        # initialize
        self.pegs[self.initial_peg].fill(self.num_discs)

        observation = self.get_state()
        return observation

    def step(self, action):
        from_peg_num = action['from_peg']
        to_peg_num = action['to_peg']
        if LOG_LEVEL == DEBUG:
            print 'moving from peg %d to peg %d' % (from_peg_num, to_peg_num)
        from_peg = self.pegs[int(from_peg_num)]
        to_peg = self.pegs[int(to_peg_num)]

        if not from_peg.discs:
            rot = self.get_state()
            rot['reward'] = -1
            rot['terminal'] = True
            if LOG_LEVEL == DEBUG:
                print "move invalid: empty"
                print "rot:", rot
            return rot

        if to_peg.discs and from_peg.discs[-1] > to_peg.discs[-1]:
            rot = self.get_state()
            rot['reward'] = -1
            rot['terminal'] = True
            if LOG_LEVEL == DEBUG:
                print "move invalid: too big"
                print "rot:", rot
            return rot

        disc = from_peg.pop()
        to_peg.push(disc)
        if LOG_LEVEL == INFO:
            print "\t".join( [ str(peg.discs) for peg in self.pegs ] )

        rot = self.get_state()
        rot['reward'] = self.get_reward()
        rot['terminal'] = self.is_final()
        if rot['terminal']: print 'FINAL...'
        return rot

    def get_reward(self):
        reward = 0
        for peg in self.pegs:
            if not peg.is_valid:
                # invalid disc layout
                if LOG_LEVEL == DEBUG:
                    print 'peg is invalid', peg.discs
                reward = -1
                break
            elif self.discs_moved(peg):
                # all discs moved to another peg
                if LOG_LEVEL == DEBUG:
                    print 'all discs moved to peg %d' % peg.number, peg.discs
                reward = 1
                break
        return reward

    def is_final(self):
        is_final = False
        for peg in self.pegs:
            is_final |= self.discs_moved(peg)
        return is_final

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


if __name__ == "__main__":
    from reply.runner import Runner
    r = Runner(HanoiAgent(), HanoiEnvironment(), Experiment())
    r.run()
