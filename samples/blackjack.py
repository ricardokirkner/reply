import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import reply
import random

# cards
SPADES, HEARTS, DIAMONDS, CLUBS = range(4)
SUITES = [SPADES, HEARTS, DIAMONDS, CLUBS]
DECK = ((number,suit) for number in xrange(1,14) for suit in SUITES)
CARDS = [card for card in DECK for deck in xrange(8)]
# actions
#HIT, STAND, SPLIT = range(4,7)
HIT, STAND = range(4,6)

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

    def setup(self):
        self.deal(self, 1)
        for player in self.players:
            self.deal(player)
        self.deal(self, 1)

    def teardown(self):
        self.players = []
        self.cards = []

    def play(self):
        self.total_points, self.soft_hand = eval_hand(self.cards)
        while self.total_points < 17:
            self.deal(self, 1)
            self.total_points, self.soft_hand = eval_hand(self.cards)



class Player(reply.World):
    def __init__(self, dealer):
        self.dealer = dealer
        self.cards = []
        self.total_points = 0
        self.soft_hand = False
        self._is_final = False
        self.bet = 0

    @staticmethod
    def get_state_space():
        return [ reply.dimension(0, 30, 31), 
                 reply.dimension(1, 13, 13),
                 reply.dimension(0, 1, 2),
                 reply.dimension(0, 1, 2)]

    @staticmethod
    def get_action_space():
        #return [ reply.dimension(HIT, SPLIT, 3) ]
        return [ reply.dimension(HIT, STAND, 2) ]

    def get_reward(self, state):
        reward = 0
        if self.is_final(state):
            if self.is_busted():
                reward = -self.bet
            else:
                self.dealer.play()
                if self.has_won():
                    reward = self.bet
                else:
                    reward = -self.bet
        return reward

    def is_busted(self):
        return self.total_points > 21

    def has_won(self):
        if self.dealer.total_points == 21:
            print 'DEALER BLACK JACK'
            return False
        elif self.dealer.total_points > 21:
            print 'DEALER BUSTED'
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

    def do_action(self, solver, action):
        if action == HIT:
            print 'HIT'
            self.dealer.deal(self, 1)
        elif action == STAND:
            print 'STAND'
            self._is_final = True
        #elif action == SPLIT:
        #    print 'SPLIT'
        #    pass
        else:
            print 'ACTION', action, 'is not available.'
            sys.exit(1)
        return self.get_state()

    def get_initial_state(self):
        return self.get_state()

    def get_state(self):
        self.total_points, self.soft_hand = eval_hand(self.cards)
        state = (self.total_points, self.dealer.cards[1][0], self.soft_hand, self.is_split_hand())
        return state

    def is_split_hand(self):
        return self.cards[-1] == self.cards[-2]

    def do_hit(self):
        return HIT
    def do_stand(self):
        return STAND
    #def do_split(self):
    #    return SPLIT
    
def eval_hand(cards):
    points = 0
    soft_hand = False
    cards.sort(lambda x, y: x[0] > y[0])
    softs = 0
    for number,suit in cards:
        if number == 1:
            softs += 1
        points += number
    # use 1 as 11 if we want to
    for item in xrange(softs):
        if points + 10 < 22:
            points += 10
            soft_hand = True
    return points, soft_hand


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
    cash = 1000
    for episode in xrange(10000):
        if cash == 0:
            break
        dealer = Dealer()
        player = Player(dealer)
        player.bet = min(5, cash)
        dealer.players = [player]
        dealer.setup()
        total_reward, steps = rl.run(player)
        dealer.teardown()
        cash += total_reward
        print 'Episode: ', episode, '  Steps: ', steps, '  Player: ', player.total_points, '  Dealer: ', dealer.total_points, '  Reward: ', total_reward, '  Cash: ', cash
