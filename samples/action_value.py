# This code is so you can run the samples without installing the package
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
#

import random

from reply.agent import LearningAgent
from reply.datatypes import Integer
from reply.encoder import SpaceEncoder, StateActionEncoder
from reply.environment import Environment
from reply.experiment import Experiment
from reply.learner import QLearner
from reply.policy import EGreedyPolicy
from reply.storage import TableStorage


class ActionValueAgent(LearningAgent):
    def init(self, task_spec):
        super(ActionValueAgent, self).init(task_spec)
        learning_rate = 1
        learning_rate_decay = 0.999
        learning_rate_min = 0.001
        random_action_rate = 1

        state_encoder = SpaceEncoder(self._observation_space)
        action_encoder = SpaceEncoder(self._action_space)
        encoder = StateActionEncoder(state_encoder, action_encoder)
        storage = TableStorage((1, 10), encoder)
        policy = EGreedyPolicy(storage, random_action_rate)
        self.learner = QLearner(policy, learning_rate, learning_rate_decay,
                                learning_rate_min)

        self.last_observation = None
        self.last_action = None


class ActionValueEnvironment(Environment):
    actions_spec = {'choice': Integer(0, 9)}
    observations_spec = {'state': Integer(1, 1)}
    problem_type = "episodic"
    discount_factor = 1.0
    rewards = Integer(-1, 1)


    def on_set_num_action(self, n):
        self.set_action_space(choice=Integer(0, n-1))

    def _init(self):
        maxval = self._action_space["choice"].max + 1
        self.ps = [ p/float(maxval) for p in range(1, maxval+1) ]

    def _start(self):
        return dict(state=0)

    def _step(self, action):
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
