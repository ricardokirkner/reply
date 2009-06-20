# This code is so you can run the samples without installing the package
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
#

import random

from reply.agent import Agent
from reply.datatypes import Integer
from reply.encoder import SpaceEncoder, StateActionEncoder
from reply.environment import Environment
from reply.experiment import Experiment
from reply.learner import QLearner
from reply.policy import EGreedyPolicy
from reply.storage import TableStorage


class StateValueAgent(Agent):
    def _init(self, task_spec):
        learning_rate = 1
        learning_rate_decay = 0.99
        learning_rate_min = 0.001
        random_action_rate = 1

        state_encoder = SpaceEncoder(self._observation_space)
        action_encoder = SpaceEncoder(self._action_space)
        encoder = StateActionEncoder(state_encoder, action_encoder)
        storage = TableStorage((10, 1), encoder)
        policy = EGreedyPolicy(storage, random_action_rate)
        self.learner = QLearner(policy, learning_rate, learning_rate_decay,
                                learning_rate_min)

        self.last_observation = None
        self.last_action = None

    def _start(self, observation):
        self.learner.new_episode()
        action = self.learner.policy.select_action(observation)
        self.last_observation = observation
        self.last_action = action
        return action

    def _step(self, reward, observation):
        self.learner.update(self.last_observation, self.last_action, reward,
                            observation)
        action = self.learner.policy.select_action(observation)
        self.last_observation = observation
        self.last_action = action
        return action

    def _end(self, reward):
        self.learner.update(self.last_observation, self.last_action, reward,
                            None)


class StateValueEnvironment(Environment):
    actions_spec = {'pass': Integer(1, 1)}
    observations_spec = {'state': Integer(0, 9)}
    problem_type = 'episodic'
    discount_factor = 1.0
    rewars = Integer(0, 1)
    state = None

    def _init(self):
        maxval = self._observation_space['state'].max + 1
        self.ps = [ p/float(maxval) for p in range(1, maxval+1) ]

    def _start(self):
        self.state = random.randint(0, len(self.ps)-1)
        return dict(state=self.state)

    def _step(self, action):
        if random.random() < self.ps[self.state]:
            r = 1
        else:
            r = 0
        rot = dict(state=self.state, reward=r, terminal=True)
        return rot

    def _end(self, reward):
        pass


if __name__ == "__main__":
    from reply.runner import Runner
    r = Runner(StateValueAgent(), StateValueEnvironment(), Experiment())
    r.run()
