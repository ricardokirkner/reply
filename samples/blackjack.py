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
    def inner(self, solver, action):
        s = self.get_state()[0]
        r = f(self, solver, action[0])
        self.history.append( (s,action) )
        return r
    return inner

class Player(reply.World):
    def __init__(self, dealer):
        self.history = []
        self.dealer = dealer
        self.cards = []
        self.total_points = 0
        self.soft_hand = False
        self._is_final = False
        self.bet = 0

    @staticmethod
    def get_state_space():
        return [ reply.dimension(2,22, 21), 
                 ]

    @staticmethod
    def get_action_space():
        return [ reply.dimension(0, 1, 2) ]

    def get_reward(self, state):
        reward = 0
        if self.is_final(state):
            if self.is_busted():
                reward = -1
            else:
                self.dealer.play()
                if self.has_won():
                    reward = +1
                else:
                    reward = -1
        return reward

    def is_busted(self):
        return self.total_points > 21

    def has_won(self):
        if self.dealer.total_points == 21:
            #print 'DEALER BLACK JACK'
            return False
        elif self.dealer.total_points > 21:
            #print 'DEALER BUSTED'
            return True
        elif self.total_points > 21:
            print 'PLAYER BUSTED'
            return False
        else:
            return self.dealer.total_points < self.total_points

    def is_final(self, state):
        total_points = state[0]
        if total_points > 21:
            return True
        else:
            return self._is_final

    @record
    def do_action(self, solver, action):
        if action == HIT:
            #print "HIT", self.get_state(), self.cards
            self.dealer.deal(self, 1)
        elif action == STAND:
            #print "STAND", self.get_state()
            self._is_final = True
        else:
            print 'ACTION', action, 'is not available.'
            sys.exit(1)
        return self.get_state()

    def get_initial_state(self):
        return self.get_state()

    def get_state(self):
        return [min(22, self.total_points)]


def eval_hand(cards):
    return sum(cards)


if __name__ == '__main__':
    space = Player.get_state_space(), Player.get_action_space()
    encoder = reply.encoder.DistanceEncoder(*space)
    storage = reply.storage.TableStorage(encoder)
    alpha = 0.9
    epsilon = 0.1
    gamma = 0.9
    alpha_decay = 1.0
    epsilon_decay = 1.0
    learner = reply.learner.QLearner(alpha, epsilon, gamma, alpha_decay)
    selector = reply.selector.EGreedySelector(epsilon, epsilon_decay)
    rl = reply.RL(learner, storage, encoder, selector)

    #dealer = Dealer()
    wins = 0.0
    #rl.storage.state[20,0] = rl.storage.state[20,1] = -10
    for episode in xrange(100000):
        dealer = Dealer()
        player = Player(dealer)
        dealer.players = [player]
        dealer.setup()
        total_reward, steps = rl.run(player)
        if total_reward > 0:
            wins += 1
        dealer.teardown()
        print 'Episode: ', episode, '  Steps: ', steps, '  Player: ', player.total_points, '  Dealer: ', dealer.total_points, '  Reward: ', total_reward, 'avg:', wins/(episode+1)
        continue
        s = rl.storage.state
        for i in range(0,21):
            print "%02d   "%(i+2), "%012f     %012f"%(s[i,0], s[i,1]), ["H","S"][( s[i, 0] < s[i, 1])]

        #raw_input("next")