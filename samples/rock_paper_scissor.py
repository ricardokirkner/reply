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

ROCK, PAPER, SCISSOR = range(3)
verbose_choices = {ROCK: "rock", PAPER: "paper", SCISSOR: "scissor"}

defeats = {ROCK: SCISSOR, PAPER: ROCK, SCISSOR: PAPER}


def eval_hand(cards):
    return sum(cards)


# Common model
observations = Space({'state': Integer(0, 0)})
actions = Space({'play': Integer(0, 2)})
rock_paper_scissor_model = Model(observations, actions)


class RockPaperScissorAgent(LearningAgent):
    model = rock_paper_scissor_model
    storage_class = TableStorage
    policy_class = EGreedyPolicy
    learner_class = QLearner

    learning_rate = 0.1
    learning_rate_decay = 1.0
    discount_value = 0.9
    random_action_rate = 0.9
    random_action_rate_decay = 0.999


class RockPaperScissorEnvironment(Environment):
    problem_type = "episodic"
    discount_factor = 0.9
    rewards = Integer(-1, 1)
    model = rock_paper_scissor_model

    def init(self):
        self.history = []
        return super(RockPaperScissorEnvironment, self).init()

    def start(self):
        self.history = []
        self.total_points = 0
        self.choice = 0
        observation = dict(state=0)
        return observation

    def step(self, action):
        other = random.randint(0, 2)
        play = action['play']
        if defeats[play] == other:
            reward = +1
        elif defeats[other] == play:
            reward = -1
        else:
            reward = 0
        rot = dict(state=0, reward=reward, terminal=True)
        return rot


if __name__ == '__main__':
    from reply.runner import Runner
    r = Runner(RockPaperScissorAgent(), RockPaperScissorEnvironment(),
               Experiment())
    r.run()
