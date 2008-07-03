# This code is so you can run the samples without installing the package
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
#

import random, cPickle

from pyglet import window
import primitives as draw
from pyglet import clock
from pyglet.gl import *

import pymunk._chipmunk as cp
import pymunk.util as u
from pymunk import Vec2d as vec2d
import ctypes 

from numpy import arange, array, size
INFINITY = float("infinity")
BACKGROUND = cp.cpBodyNew(INFINITY, INFINITY)

import reply

def cpvrotate(v1, v2):
    return vec2d(v1.x*v2.x - v1.y*v2.y, v1.x*v2.y + v1.y*v2.x)
def cpvadd(v1, v2):
    return vec2d(v1.x + v2.x, v1.y + v2.y)
dt = 1.0/30.0


class Ball:
    def __init__(self, space, mass, radius, pos=(300,300), color=(0.,.9,0.,1.)):
        self.color = color
        self.mass = mass
        self.radius = radius
        ball_moment = cp.cpMomentForCircle(mass, 5, 0.0, vec2d(0,0))
        self.body = cp.cpBodyNew( mass, ball_moment )
        self.shape = cp.cpCircleShapeNew(self.body, 
                radius, vec2d(0.0, 0.0) )
        self.shape.contents.e = 0.5
        self.set_position( pos )
        cp.cpSpaceAddBody(space, self.body)
        cp.cpSpaceAddShape( space, self.shape )
        
    def set_position(self, position):
        self.body.contents.p = vec2d(position[0], position[1])

    def render(self):
        p = self.body.contents.p
        c = draw.Circle(p[0], p[1] ,width=self.radius*2,color=self.color, style=GLU_LINE, stroke=1)
        c.render()
        
class Wall:
    def __init__(self, space, start, end, color=(0.,1.,1.,1.), radius=50):
        self.color = color
        self.start = start
        self.end = end
        self.radius = radius
        self.shape = cp.cpSegmentShapeNew(BACKGROUND, 
            vec2d(*start), vec2d(*end), radius)
        self.shape.contents.e = 0.1
        self.shape.contents.u = 0.2
        cp.cpSpaceAddStaticShape(space, self.shape)
        self.line = draw.Line(
            (start[0], start[1]+radius),
            (end[0], end[1]+radius)
            ,stroke=2,color=color)
        
    def render(self):
        self.line.render()
        
class Box:
    def __init__(self, space, mass, width, height, pos, color=(0,1,1,1)):
        poly = [
            [ width/2, height/2 ],
            [ width/2, -height/2 ],
            [ -width/2, -height/2 ],
            [ -width/2, height/2 ],
            ]
        p_num = len(poly)
        P_ARR = vec2d * p_num
        p_arr = P_ARR(vec2d(0,0))
        for i, (x,y) in enumerate( poly ):
            p_arr[i].x = x
            p_arr[i].y = y
                       
        moment = cp.cpMomentForPoly( mass, p_num, p_arr, vec2d(0,0))
        self.color = color
        self.body = cp.cpBodyNew( mass, moment )
        self.body.contents.p = vec2d( *pos )
        cp.cpSpaceAddBody( space, self.body )
        self.shape = shape = cp.cpPolyShapeNew(
            self.body, p_num, p_arr, vec2d(0,0)
            )
        shape.contents.u = 0.5
        cp.cpSpaceAddShape(space, shape)
    
        
    def render(self):
        p = self.body.contents.p
        body = self.shape.contents.body
        shape = ctypes.cast(self.shape, ctypes.POINTER(cp.cpPolyShape))
        num = shape.contents.numVerts
        verts = shape.contents.verts
        ps = [
            cpvadd(body.contents.p, 
                cpvrotate(verts[i], body.contents.rot))
            for i in range(num)]
        for i,p in enumerate(ps):
            draw.Line(
                (p.x, p.y),
                (ps[(i+1)%num].x, ps[(i+1)%num].y),
                stroke=1,color=self.color
            ).render()

           

class Pendulum(reply.World):
            
    def __init__(self, filename):
        self.filename = filename
        try:
            pend = cPickle.load(open(filename))
        except IOError:
            e = reply.encoder.DistanceEncoder(
                self.get_state_space(), self.get_action_space()
                )
            pend = reply.RL(
                    reply.learner.QLearner(0.1, 0.2,1,0.05),
                    reply.storage.TableStorage(e),
                    e, 
                    reply.selector.EGreedySelector(0.1, 1)
                )
        self.rl = pend
        clock.set_fps_limit(30)
        self.win = window.Window()

        cp.cpInitChipmunk()


    def get_action_space(self):
        return [reply.dimension(-1,1,20)]
        
    def get_state_space(self):
        return [
            # base velocity
            reply.dimension(-1000,1000,100),
            # ball relative x position
            reply.dimension(-84,84,30),
            # ball x velocity
            reply.dimension(-300,300,10),
            # ball y velocity
            reply.dimension(-300,300,10),
        ] 
    
    def get_reward(self, s ):
        died = 0
        if self.ball.body.contents.p.y < 100:
            died = -1000
        return (self.ball.body.contents.p.y-170 + died)
        
    def is_final(self, state):
        if self.ball.body.contents.p.y < 100:
            return True
        return False
        
    def get_state(self):
        return [
            self.base.body.contents.v.x,
            self.ball.body.contents.p.x-self.base.body.contents.p.x,
            self.ball.body.contents.v.x,
            self.ball.body.contents.v.y,
            ]
            
    def do_action(self, solver, action):
        # returns new state
        #print "DO:", action
        action = action[0]
        cp.cpBodyApplyForce(self.base.body, vec2d(action*20000,0), vec2d(0,0))
        cp.cpSpaceStep(self.space, dt+random.random()*1.0/100.0)
        cp.cpBodyResetForces(self.base.body)
        
        clock.tick()
        self.win.dispatch_events()
       

        self.win.clear()
        self.ball.render()
        self.floor.render()
        self.base.render()
        
        self.win.flip()
        
            
    def get_initial_state(self):
        # return first state
        self.space = space = cp.cpSpaceNew()
        space.contents.gravity = vec2d(0.0, -900.0)

        cp.cpSpaceResizeStaticHash(space, 50.0, 200)
        cp.cpSpaceResizeActiveHash(space, 50.0, 10)
      
        self.ball = ball = Ball(space, 5, 10, pos=(310, 180))
        self.floor = Wall(space, (-6000,40), (6000,40))
        self.base = base = Box(space, 5, 50, 20, (300,100))
        cp.cpResetShapeIdCounter()
        joint = cp.cpPinJointNew(
            ball.body, base.body, vec2d(0,0), vec2d(0,0)
            )
        cp.cpSpaceAddJoint(space, joint)
        
        return self.get_state()
        
    def cleanUp(self):
        cp.cpSpaceFree( self.space )
        
    def learn(self, iterations, max_steps=10000):
        try:
            for episode in xrange(iterations):
                self.rl.new_episode(self)
                for step in range(max_steps):
                    if not self.rl.step(self):
                        break
                print 'Espisode: ',episode,'  Steps:',step,'  Reward:',self.rl.total_reward, 'alpha', self.rl.learner.alpha, 'epsilon', self.rl.selector.epsilon
        except KeyboardInterrupt:
            cPickle.dump(self.policy, open(self.filename, "w"))              
        else:    
            cPickle.dump(self.policy, open(self.filename, "w"))           

def run(maxepisodes):
    p = Pendulum("pendulum.pkl")
    p.learn(maxepisodes)    
                 

if __name__ == '__main__':
    run(1000000)    

  
