import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random

from reply.agent import LearningAgent
from reply.datatypes import Integer, Model, Space
from reply.encoder import SpaceEncoder
from reply.environment import Environment
from reply.experiment import Experiment
from reply.learner import QLearner
from reply.policy import EGreedyPolicy
from reply.storage import TableStorage

# cards
DECK = [ number for number in range(1,10)+[10,10,10] for suit in range(4) ]
CARDS = [card for card in DECK for deck in xrange(8)]
# actions
HIT, STAND = range(2)

DEBUG = False


def eval_hand(cards):
    return sum(cards)

def record(f):
    def inner(self, action):
        s = self.get_state()
        r = f(self, action)
        self.history.append( (s,action) )
        return r
    return inner


class Dealer(object):
    def __init__(self):
        # deck only contains 80% of available cards
        random.shuffle(CARDS)
        limit = int(0.8*len(CARDS))
        self.deck = CARDS[:limit]
        self.total_points = 0
        self.soft_hand = False
        self.cards = []
        self.players = []

    def deal(self, player, cards=2):
        for card in xrange(cards):
            player.cards.append(self.deck.pop())
            player.total_points = eval_hand( player.cards )

    def setup(self):
        self.deal(self, 1)
        for player in self.players:
            self.deal(player)
        self.deal(self, 1)

    def teardown(self):
        self.players = []
        self.cards = []

    def play(self):
        self.total_points = eval_hand(self.cards)
        while self.total_points < 17:
            self.deal(self, 1)
            self.total_points = eval_hand(self.cards)


# Common model
observations = Space({'total_points': Integer(2, 22)})
actions = Space({'play': Integer(0, 1)})
blackJackModel = Model(observations, actions)


class BlackJackEncoder(SpaceEncoder):
    def encode(self, item):
        encoded_item = [item[key] - self.space[key].min \
                        for key in self.space.get_names_list()]
        return tuple(encoded_item)


class BlackJackAgent(LearningAgent):
    model = blackJackModel
    state_encoder_class = BlackJackEncoder
    action_encoder_class = SpaceEncoder
    storage_class = TableStorage
    policy_class = EGreedyPolicy
    learner_class = QLearner

    learning_rate = 0.1
    learning_rate_decay = 1.0
    discount_value = 0.9
    random_action_rate = 0.1
    random_action_decay = 0.999

    def start(self, observation):
        #print 'NEW EPISODE'
        return super(BlackJackAgent, self).start(observation)

    def end(self, reward):
        super(BlackJackAgent, self).end(reward)
        #print self.learner.policy.storage.data


class BlackJackEnvironment(Environment):
    problem_type = "episodic"
    discount_factor = 1.0
    rewards = Integer(-1, 1)
    model = blackJackModel

    def init(self):
        super(BlackJackEnvironment, self).init()
        self.history = []

    def start(self):
        #print self.history
        self.history = []
        self.cards = []
        self.total_points = 0
        self.final = False
        self.soft_hand = False
        self.bet = 0

        self.dealer = Dealer()
        self.dealer.players = [self]
        self.dealer.setup()

        observation = self.get_state()
        return observation

    @record
    def step(self, action):
        play = action['play']
        if play == HIT:
            if DEBUG:
                print "HIT", self.get_state(), self.cards
            self.dealer.deal(self, 1)
        elif play == STAND:
            if DEBUG:
                print "STAND", self.get_state()
            self.final = True
        else:
            if DEBUG:
                print 'ACTION', play, 'is not available.'
            sys.exit(1)

        reward = self.get_reward()
        terminal = self.is_final()
        rot = dict(total_points=self.total_points, reward=reward,
                   terminal=terminal)
        return rot

    def cleanup(self):
        self.dealer.teardown()

    def is_final(self):
        state = self.get_state()
        return self.final or state['total_points'] == 22

    def get_state(self):
        observation = dict(total_points=min(22, self.total_points))
        return observation

    def get_reward(self):
        reward = 0
        if self.is_final():
            if self.total_points > 21:
                reward = -1
            else:
                self.dealer.play()
                if self.has_won():
                    reward = +1
                else:
                    reward = -1
        return reward

    def has_won(self):
        if self.dealer.total_points == 21:
            if DEBUG:
                print 'DEALER BLACK JACK'
            return False
        elif self.total_points > 21:
            if DEBUG:
                print 'PLAYER BUSTED'
            return False
        elif self.dealer.total_points > 21:
            if DEBUG:
                print 'DEALER BUSTED'
            return True
        else:
            return self.dealer.total_points < self.total_points


if __name__ == '__main__':
    from reply.runner import Runner
    r = Runner(BlackJackAgent(), BlackJackEnvironment(), Experiment())
    r.run()
