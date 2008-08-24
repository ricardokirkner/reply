import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import reply
import random

PIEDRA, PAPEL, TIJERA = range(3)
verbose_choices = {PIEDRA: "piedra", PAPEL: "papel", TIJERA: "tijera"}

defeats = {PIEDRA: TIJERA, PAPEL: PIEDRA, TIJERA: PAPEL}


class Player(reply.World):
    def __init__(self):
        self.history = []
        self.total_points = 0
        self.choice = 0

    @staticmethod
    def get_state_space():
        return [ reply.dimension(0, 0, 1),
                 ]

    @staticmethod
    def get_action_space():
        return [ reply.dimension(0, 2, 3) ]

    def get_reward(self, state):
        other = random.randint(0, 2)
        if defeats[self.choice] == other:
            reward = +1
        elif defeats[other] == self.choice:
            reward = -1
        else:
            reward = 0
        return reward


    def is_final(self, state):
        return True

    def do_action(self, solver, action):
        self.choice = action[0]
        return self.get_state()

    def get_initial_state(self):
        return self.get_state()

    def get_state(self):
        return [0]


def eval_hand(cards):
    return sum(cards)


if __name__ == '__main__':
    space = Player.get_state_space(), Player.get_action_space()
    encoder = reply.encoder.DistanceEncoder(*space)
    storage = reply.storage.TableStorage(encoder)
    alpha = 0.1
    epsilon = 0.9
    gamma = 0.9
    alpha_decay = 1.0
    epsilon_decay = 0.999
    learner = reply.learner.QLearner(alpha, gamma, alpha_decay)
    selector = reply.selector.EGreedySelector(epsilon, epsilon_decay)
    rl = reply.RL(learner, storage, encoder, selector)

    wins = 0.0
    time_avg = 1.0/500
    for episode in xrange(3000):
        player = Player()
        total_reward, steps = rl.run(player)
        if total_reward > 0:
            wins = time_avg + (1-time_avg) * wins
        else:
            wins = (1-time_avg) * wins

        print 'Episode: ', episode, '  Steps: ', steps, '  Reward: ', total_reward, 'avg:', wins, 'epsilon:', selector.epsilon
        print rl.storage.state

