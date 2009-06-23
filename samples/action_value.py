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
observations = Space({'state': Integer(1, 1)})
actions = Space({'choice': Integer(0, 9)})
actionValueModel = Model(observations, actions)


class ActionValueAgent(LearningAgent):
    model = actionValueModel
    state_encoder_class = SpaceEncoder
    action_encoder_class = SpaceEncoder
    storage_class = TableStorage
    policy_class = EGreedyPolicy
    learner_class = QLearner

    learning_rate = 1
    learning_rate_decay = 0.999
    learning_rate_min = 0.001
    random_action_rate = 1


class ActionValueEnvironment(Environment):
    problem_type = "episodic"
    discount_factor = 1.0
    rewards = Integer(-1, 1)
    model = actionValueModel

    def init(self):
        maxval = self.model.actions["choice"].max + 1
        self.ps = [ p/float(maxval) for p in range(1, maxval+1) ]

    def start(self):
        return dict(state=0)

    def step(self, action):
        if random.random() < self.ps[action['choice']]:
            r = 1
        else:
            r = 0
        rot = dict(state=0, reward=r, terminal=True)
        return rot


if __name__=="__main__":
    from reply.runner import Runner
    r = Runner(ActionValueAgent(), ActionValueEnvironment(), Experiment())
    r.run()
