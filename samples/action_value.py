# This code is so you can run the samples without installing the package
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
#

import random

from reply.agent import Agent
from reply.environment import Environment
from reply.experiment import Experiment
from reply.types import Integer


class ActionValueAgent(Agent):
    def _init(self, task_spec):
        pass

    def _start(self, observation):
        pass

    def _step(self):
        pass

class ActionValueEnvironment(Environment):
    action_space = dict(choice=Integer(0, 10))
    observation_space = dict(state=Integer(0, 0))
    problem_type = "episodic"
    discount_factor = 1.0
    rewards = (-1, 1)

    def on_set_num_action(self, n):
        self.set_action_space(choice=Integer(0, n-1))

    def _init(self):
        maxval = self.action_space["choice"].max
        self.ps = [ p/float(maxval) for p in range(maxval) ]

    def _start(self):
        return dict(state=0)

    def _step(self, action):
        if random.random() > self.ps[action.choice]:
            r = 1
        else:
            r = 0
        return dict(state=0, reward=r, final=True)


class ActionValueExperiment(Experiment):
    def _init(self):
        pass

    def _start(self):
        pass

    def _step(self):
        pass

    def _cleanup(self):
        pass


if __name__=="__main__":
    import sys
    def usage():
        print "%s [agent|environment|experiment]" % sys.argv[0]

    if len(sys.argv) < 2:
        usage()
        exit(0)

    role = sys.argv[1]
    if role == 'agent':
        from reply.glue import start_agent
        start_agent(ActionValueAgent())
    elif role == 'environment':
        from reply.glue import start_environment
        start_environment(ActionValueEnvironment())
    elif role == 'experiment':
        from reply.glue import start_experiment
        start_experiment(ActionValueExperiment())
    else:
        usage()
        exit(0)

