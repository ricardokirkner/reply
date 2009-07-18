import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random


from reply.agent import LearningAgent
from reply.datatypes import Integer, Model, Space
from reply.environment import Environment
from reply.experiment import Experiment
from reply.learner import QLearner
from reply.policy import EGreedyPolicy
from reply.storage import TableStorage

# Common model
max_money = 100
observations = Space({'money': Integer(0, max_money)})
actions = Space({'bet': Integer(1, max_money)})
GamblerModel = Model(observations, actions)

class GamblerAgent(LearningAgent):
    model = GamblerModel
    storage_class = TableStorage
    policy_class = EGreedyPolicy
    learner_class = QLearner

    learning_rate = 1
    learning_rate_decay = 0.999999
    learning_rate_min = 0.001
    random_action_rate = 0.01

    def start(self, observation):
        #print 'NEW EPISODE'
        return super(GamblerAgent, self).start(observation)

    def end(self, reward):
        super(GamblerAgent, self).end(reward)
        #print self.learner.policy.storage.data

    def on_get_state(self):
        return self.learner.policy.storage.data.tolist()

class GamblerEnvironment(Environment):
    problem_type = "episodic"
    discount_factor = 1.0
    rewards = Integer(-1, 1)
    model = GamblerModel

    win_probability = 0.4

    def init(self):
        return super(GamblerEnvironment, self).init()

    def start(self):
        #print self.history
        self.money = random.randint(1,max_money-1)

        return dict(money=self.money)

    def step(self, action):
        bet = min(self.money, action['bet'])
        if random.random() > self.win_probability:
            self.money = min(self.money + bet, max_money)
        else:
            self.money -= bet
        if self.money == max_money:
            reward = 1
            terminal = True
        elif self.money == 0:
            reward = 0
            terminal = True
        else:
            reward = 0
            terminal = False
        rot = dict(money=self.money, reward=reward,
                   terminal=terminal)
        return rot

def get_policy(data):
    winners = []
    #import numpy
    #data = numpy.array(data)
    for r, row in enumerate(data):
        m = max(row)
        i = min(r, list(row).index(m))
        winners.append(i)
    return winners

class GamblerExperiment(Experiment):
    model = GamblerModel
    max_episodes = 200000000

    def run(self):
        self.init()
        win = 0
        import pylab
        pylab.ion()
        line = pylab.plot([100]*50+[0]*51)[0]
        for episode in xrange(self.max_episodes):
            self.start()
            steps = 0
            terminal = False
            while steps < self.max_steps and not terminal:
                roat = self.step()
                terminal = roat["terminal"]
                steps += 1
            win += (self.return_reward()-win)/2000.0
            new_data = get_policy(self.agent_call("get_state"))
            if episode % 1000 == 0:
                print episode
                line.set_ydata(new_data)
                pylab.draw()
        pylab.show()
        self.cleanup()

def main():
    from reply.runner import Runner
    r = Runner(GamblerAgent(), GamblerEnvironment(), GamblerExperiment())
    r.run()
if __name__ == '__main__':
    main()
