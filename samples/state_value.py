# This code is so you can run the samples without installing the package
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
#

import random

from reply.agent import LearningAgent
from reply.datatypes import Integer, Model, Space
from reply.encoder import SpaceEncoder
from reply.environment import Environment
from reply.experiment import Experiment
from reply.learner import QLearner
from reply.policy import EGreedyPolicy
from reply.storage import TableStorage


# Common model
observations = Space({'state': Integer(0, 9)})
actions = Space({'pass': Integer(1, 1)})
stateValueModel = Model(observations, actions)


class StateValueAgent(LearningAgent):
    model = stateValueModel
    state_encoder_class = SpaceEncoder
    action_encoder_class = SpaceEncoder
    storage_class = TableStorage
    policy_class = EGreedyPolicy
    learner_class = QLearner

    learning_rate = 1
    learning_rate_decay = 0.99
    learning_rate_min = 0.001
    random_action_rate = 1


class StateValueEnvironment(Environment):
    problem_type = 'episodic'
    discount_factor = 1.0
    rewards = Integer(0, 1)
    model = stateValueModel

    def init(self):
        maxval = self.model.observations['state'].max + 1
        self.ps = [ p/float(maxval) for p in range(1, maxval+1) ]
        self.state = None

    def start(self):
        self.state = random.randint(0, len(self.ps)-1)
        return dict(state=self.state)

    def step(self, action):
        if random.random() < self.ps[self.state]:
            r = 1
        else:
            r = 0
        rot = dict(state=self.state, reward=r, terminal=True)
        return rot


if __name__ == "__main__":
    from reply.runner import Runner
    r = Runner(StateValueAgent(), StateValueEnvironment(), Experiment())
    r.run()
