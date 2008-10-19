# This code is so you can run the samples without installing the package
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
#

import reply
import numpy
from numpy import array
import random

from pyglet import window
from pyglet import clock
from pyglet.gl import *

class Gambler(reply.World):
    def __init__(self, rl):
        " p == win probability "
        super(Gambler, self).__init__(rl)
        self.p = getattr(rl, "win_probability")

    def is_final(self):
        if self.cash <= 0  or self.cash >= 20:
            return True
        return False

    def do_action(self, action):
        stake = action[0]
        if stake > self.cash:
            raise reply.ActionNotPossible()
        if random.random() < self.p:
            self.cash += stake
        else:
            self.cash -= stake
        self.cash = max(0,self.cash)
        self.cash = min(20, self.cash)

    def new_episode(self):
        self.cash = random.randint(1,19)

    def get_state(self):
        return dict(cash=self.cash)

class GamblerAgent(reply.LearningAgent):
    win_probability = 0.4
    learning_rate = 1
    learning_rate_decay = 0.99
    learning_rate_min = 0.001

    random_action_rate = 1

    world_class = Gambler
    learner_class = reply.learner.QLearner
    selector_class = reply.selector.EGreedySelector
    storage_class = reply.storage.TableStorage
    encoder_class = reply.encoder.DistanceEncoder

    def get_action_space(self):
        return [ reply.Dimension("bet", 1,10,10) ]

    def get_state_space(self):
        return [ reply.Dimension("cash", 0,20,21) ]

    def get_reward(self):
        if self.world.cash >= 20:
            return 1
        if self.world.cash <= 0:
            return -1
        return 0

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
    reply.Experiment(GamblerAgent()).run()
