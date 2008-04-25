import rl
import numpy
from numpy import array
import random
import plot

from pyglet import window
from pyglet import clock
from pyglet.gl import *

class StateValue(rl.World):
    def __init__(self, ps):
        " p == win probability "
        self.ps = ps

    def get_action_space(self):
        return [ rl.dimension(1,1,1) ]
    
    def get_problem_space(self):
        return [ rl.dimension(0,len(self.ps)-1,len(self.ps)) ]
        
    def get_reward(self, state):
        if self.won:
            return 1
        return 0
        
    def is_final(self, state):
        return True
        
    def do_action(self, solver, action):

        selection = action[0]
        if random.random() < self.ps[int(self.state)]:
            self.won = True
        return [self.state]
        
    def get_initial_state(self):
        self.won = 0
        self.state = random.randint(1,len(self.ps)-1)
        return [self.state]
        
    
class VisualStateValue(StateValue):
    def __init__(self, p):
        super(VisualStateValue, self).__init__(p)        
        self.x_size = 400
        self.y_size = 400
        self.win = window.Window(self.x_size, self.y_size)
        self.last_episode = 0
        
        self.value_plot = plot.Plot(0,0,400,390, [0,len(p)], [-1,1])
                
      
    def do_action(self, solver, action):
        next_state = super(VisualStateValue, self).do_action(self, action)
        
        self.win.dispatch_events()
        
        if solver.episodes <= self.last_episode+100:
            return next_state
        self.last_episode = solver.episodes
        self.win.clear()
        print solver.learner.storage.state
        self.value_plot.start_plot()
        # draw value for state (green)
        for i, s in enumerate(self.get_problem_space()[0]):
            e =     solver.encoder.encode_state( [ s ] )
            v = solver.learner.storage.get_max_value( 
                    solver.encoder.encode_state( [ s ] )
                )
            self.value_plot.point( i, v )
        self.win.flip()
        return next_state

if __name__ == "__main__":
    ps = [ p/10.0 for p in range(10) ]
    g = VisualStateValue(ps)
    e = rl.DistanceEncoder(g)
    r = rl.RL(
            rl.QLearner(rl.TableStorage(e), 0.001, 0.01,1),
            e, 
            rl.EGreedySelector(0.1, 1)
        )
        
    for episode in range(10000000):    
        total_reward,steps  = r.new_episode(g)
        
        print 'Espisode: ',episode,'  Steps:',steps,'  Reward:',total_reward,' epsilon: ',r.selector.epsilon, "alpha:", r.learner.alpha
