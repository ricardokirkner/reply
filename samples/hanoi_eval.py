import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reply.agent import Agent
from reply.policy import GreedyPolicy

from hanoi import HanoiAgent, HanoiEnvironment, HanoiExperiment
from hanoi import *


if __name__ == '__main__':
    from reply.runner import Runner
    learner = HanoiAgent()
    learner = learner.load()
    if learner is not None:
        agent = Agent()
        agent.storage = learner.storage
        agent.model = learner.model
        agent.policy = GreedyPolicy(agent)
        agent.initialized = True
        r = Runner(agent, HanoiEnvironment(), HanoiExperiment())
        r.run()

