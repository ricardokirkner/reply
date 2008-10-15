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
    def __init__(self, rl):
        " ps == win probability array for each action (implies number of actions)"
        super(ActionValue, self).__init__(rl)
        self.ps = getattr(rl, "action_win_probability_array")

    def is_final(self):
        return True
        
    def do_action(self, solver, action):
        selection = action[0]
        self.choice = action
        if random.random() < self.ps[int(selection)]:
            self.won = True
        
    def new_episode(self):
        self.won = 0
    
    def end_episode(self):
        err = 0
        for i,v in enumerate(self.rl.storage.state[0]):
            err += abs(self.ps[i]-v)
        self.rl.current_episode.error = err
        
    def get_state(self):
        return dict(
                state=0
            )
        
class ActionValueAgent(reply.LearningAgent):
    action_win_probability_array = [ p/10.0 for p in range(10) ]
    
    learning_rate = 1
    learning_rate_decay = 0.99
    learning_rate_min = 0.001
    
    random_action_rate = 1
    
    world_class = ActionValue
    learner_class = reply.learner.QLearner
    selector_class = reply.selector.EGreedySelector
    storage_class = reply.storage.TableStorage
    encoder_class = reply.encoder.DistanceEncoder
    
    def get_action_space(self):
        return [ reply.Dimension("choice", 0, len(self.action_win_probability_array)-1), ]
    
    def get_state_space(self):
        return [ reply.Dimension("state", 1,1) ]
        
    def get_reward(self):
        if self.world.won:
            return 1
        return 0
    
if __name__ == "__main__":
    reply.Experiment(ActionValueAgent()).run()