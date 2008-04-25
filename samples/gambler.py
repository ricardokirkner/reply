import rl
import numpy
from numpy import array
import random

from pyglet import window
from pyglet import clock
from pyglet.gl import *

class Gambler(rl.World):
    def __init__(self, p):
        " p == win probability "
        self.p = p

    def get_action_space(self):
        return [ rl.dimension(1,10,10) ]
    
    def get_problem_space(self):
        return [ rl.dimension(0,20,21) ]
        
    def get_reward(self, state):
        if state[0] >= 20:
            return 1
        #if state[0] <= 0:
        #    return -1
        return 0
        
    def is_final(self, state):
        if state[0] <= 0  or state[0] >= 20:
            return True
        return False             
        
    def do_action(self, solver, action):
        stake = action[0]
        if stake > self.capital:
            raise rl.ActionNotPossible()
        if random.random() < self.p:
            self.capital += stake
        else:
            self.capital -= stake
        self.capital = max(0,self.capital)
        self.capital = min(20, self.capital)
        #print "(", stake, ") ->", self.capital
        return [self.capital]
        
    def get_initial_state(self):
        self.capital = random.randint(1,4)
        return [self.capital]
        
    
class VisualGambler(Gambler):
    def __init__(self, p):
        super(VisualGambler, self).__init__(p)
        self.x_size = 400
        self.y_size = 400
        self.win = window.Window(self.x_size, self.y_size)
        self.last_episode = 0
        
    def do_point(self, (lx, ly), (x, y), color):
        """ draw a point x e [0,99], y e [0,1] """
        x_scale = self.x_size / len(self.get_problem_space()[0])
        y_scale = self.y_size
        glColor3f(*color)
        glBegin(GL_LINES);
        glVertex2f((lx+1)*x_scale, ly*y_scale)
        glVertex2f(x*x_scale, y*y_scale)
        glVertex2f(x*x_scale, y*y_scale) 
        glVertex2f((x+1)*x_scale, y*y_scale)
        glEnd( );
      
    def do_action(self, solver, action):
        next_state = super(VisualGambler, self).do_action(self, action)
        
        self.win.dispatch_events()
        
        if solver.episodes <= self.last_episode+100:
            return next_state
        self.last_episode = solver.episodes
        self.win.clear()
        #print solver.learner.storage.state[5]
        # draw value for state (green)
        last = -1,0.5
        for i, s in enumerate(self.get_problem_space()[0]):
            e =     solver.encoder.encode_state( [ s ] )
            v = solver.learner.storage.get_max_value( 
                    solver.encoder.encode_state( [ s ] )
                )
            #print "value", i, v
            point =  (i, v/2.0+0.5)
            self.do_point( last, point, (0,1,0,0) )
            last = point
        last = -1,0
        # draw stake for state (red)
        for i, s in enumerate(self.get_problem_space()[0]):
            a = numpy.argmax( solver.learner.storage.get_state_values( 
                        solver.encoder.encode_state( [ s ] )
                    )
                )
            point = (i, a/20.0)
            self.do_point( last, point, (1,0,0,0) )
            last = point
            
        self.win.flip()
        return next_state
        
if __name__ == "__main__":
    g = VisualGambler(0.4)
    e = rl.DistanceEncoder(g)
    r = rl.RL(
            rl.QLearner(rl.TableStorage(e), 0.9, 0.8, 0.999, 0.0001 ),
            e, 
            rl.EGreedySelector(0.5, 1)
        )
        
    for episode in range(10000000):    
        total_reward,steps  = r.new_episode(g)
        
        print 'Espisode:',episode,'  Steps:',steps,'  Reward:',total_reward,' epsilon:',r.selector.epsilon, "alpha:", r.learner.alpha
        