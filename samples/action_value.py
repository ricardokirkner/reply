# This code is so you can run the samples without installing the package
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
#

import reply
import numpy
from numpy import array
import random

class ActionValue(reply.World):
    def __init__(self, ps):
        " ps == win probability array for each action (implies number of actions)"
        self.ps = ps

    def get_action_space(self):
        return [ reply.dimension(0,len(self.ps)-1,len(self.ps))  ]
    
    def get_state_space(self):
        return [ reply.dimension(1,1,1) ]
        
    def get_reward(self, state):
        if self.won:
            return 1
        return 0
        
    def is_final(self, state):
        return True
        
    def do_action(self, solver, action):

        selection = action[0]
        if random.random() < self.ps[int(selection)]:
            self.won = True
        return [0]
        
    def get_initial_state(self):
        self.won = 0
        return [0]
    def get_state(self):
        return [0]
        
    
if __name__ == "__main__":
    ps = [ p/10.0 for p in range(10) ]
    g = ActionValue(ps)
    e = reply.encoder.DistanceEncoder(g.get_state_space(), g.get_action_space())
    r = reply.RL(
            reply.learner.QLearner(reply.storage.TableStorage(e), 0.001, 0.01,1),
            e, 
            reply.selector.EGreedySelector(0.9, 1)
        )
        
    for episode in range(1000):    
        total_reward,steps  = r.run(g)
        err = 0
        for i,v in enumerate(r.learner.storage.state[0]):
            err += abs(ps[i]-v)
        print 'Espisode: ',episode,'  Steps:',steps,'  Reward:',total_reward,' error:',err
