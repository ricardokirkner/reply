import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import reply
import random

# cards
DECK = [ number for number in range(1,10)+[10,10,10] for suit in range(4) ]
CARDS = [card for card in DECK for deck in xrange(8)]
# actions
HIT, STAND = range(2)

DEBUG = True

class Dealer:
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

def record(f):
    def inner(self, action):
        s = self.get_state()
        r = f(self, action)
        self.history.append( (s,action) )
        return r
    return inner

class BlackJack(reply.World):
    def __init__(self, rl):
        super(BlackJack, self).__init__(rl)
        self._is_final = False

    def is_final(self):
        state = self.get_state()
        return self._is_final or state['total_points'] == 22

    @record
    def do_action(self, action):
        if action == HIT:
            if DEBUG:
                print "HIT", self.get_state(), self.cards
            self.dealer.deal(self, 1)
        elif action == STAND:
            if DEBUG:
                print "STAND", self.get_state()
            self._is_final = True
        else:
            if DEBUG:
                print 'ACTION', action, 'is not available.'
            sys.exit(1)
        return self.get_state()

    def get_state(self):
        #if self._is_final:
        #    return {'total_points': 22}
        return {'total_points': min(22, self.total_points)}

    def new_episode(self):
        self.history = []
        self.cards = []
        self.total_points = 0
        self.soft_hand = False
        self.bet = 0

        self.dealer = Dealer()
        self.dealer.players = [self]
        self.dealer.setup()

    def end_episode(self):
        self.dealer.teardown()


def eval_hand(cards):
    return sum(cards)


class BlackJackAgent(reply.LearningAgent):
    learning_rate = 0.1
    learning_rate_decay = 1.0
    discount_value = 0.9

    random_action_rate = 0.1
    random_action_decay = 0.999

    world_class = BlackJack
    encoder_class = reply.encoder.DistanceEncoder
    storage_class = reply.storage.TableStorage
    learner_class = reply.learner.QLearner
    selector_class = reply.selector.EGreedySelector

    def get_state_space(self):
        return [ reply.Dimension('total_points', 2, 22) ]

    def get_action_space(self):
        return [ reply.Dimension('action', 0, 1) ]

    def get_reward(self):
        reward = 0
        if self.world.is_final():
            if self.world.total_points > 21:
                reward = -1
            else:
                self.world.dealer.play()
                if self.has_won():
                    reward = +1
                else:
                    reward = -1
        return reward

    def has_won(self):
        if self.world.dealer.total_points == 21:
            if DEBUG:
                print 'DEALER BLACK JACK'
            return False
        elif self.world.total_points > 21:
            if DEBUG:
                print 'PLAYER BUSTED'
            return False
        elif self.world.dealer.total_points > 21:
            if DEBUG:
                print 'DEALER BUSTED'
            return True
        else:
            return self.world.dealer.total_points < self.world.total_points



if __name__ == '__main__':
    agent = BlackJackAgent()

    experiment = reply.Experiment(agent)
    experiment.run()

