# This code is so you can run the samples without installing the package
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
#

from reply.environment import Environment, start
from reply.types import Integer
from reply.util import message

class ActionValueEnvironment(Environment):
    action_space = dict(choice=Integer(0,10))
    observation_space = dict(state=Integer(0,0))
    problem_type = "episodic"
    discount_factor = 1.0
    rewards = -1, 1
    
    def on_set_num_action(self, n):
        self.set_action_space(choice=Integer(0, n-1))
        
    def init(self):
        maxval = self.action_space["action"].max
        self.ps = [ p/float(maxval) for p in range(maxval) ]
        
    def start(self):
        return dict(state=0)
        
    def step(self, action):
        if random.random() > self.ps[action.choice]:
            r = 1
        else:
            r = 0
        return dict(state=0, reward=r, final=True)
    
if __name__=="__main__":
    from reply.glue import start_environment
    start_environment(ActionValueEnvironment())
	