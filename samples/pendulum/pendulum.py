# This code is so you can run the samples without installing the package
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
#

import random, cPickle
import math

from pyglet import window
from pyglet import clock
from pyglet.gl import *

import pymunk._chipmunk as cp
import pymunk.util as u
from pymunk import Vec2d as vec2d
import ctypes

from numpy import arange, array, size
import matplotlib
matplotlib.use("cairo")
import matplotlib.pyplot
import numpy


import primitives as draw


INFINITY = float("infinity")
BACKGROUND = cp.cpBodyNew(INFINITY, INFINITY)

import reply

def cpvrotate(v1, v2):
    return vec2d(v1.x*v2.x - v1.y*v2.y, v1.x*v2.y + v1.y*v2.x)
def cpvadd(v1, v2):
    return vec2d(v1.x + v2.x, v1.y + v2.y)
dt = 1.0/30.0


class Ball:
    def __init__(self, space, mass, radius, pos=(300,300), color=(0,240,0,255)):
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
        c = draw.circle((p[0]%800, p[1]), self.radius ,self.color)
        c.draw(GL_TRIANGLE_FAN)

    def __str__(self):
        return "<ball %s>"%self.body.contents.p

class Wall:
    def __init__(self, space, start, end, color=(0,255,255,255), radius=50):
        self.color = color
        self.start = start
        self.end = end
        self.radius = radius
        self.shape = cp.cpSegmentShapeNew(BACKGROUND,
            vec2d(*start), vec2d(*end), radius)
        self.shape.contents.e = 0.1
        self.shape.contents.u = 0.2
        cp.cpSpaceAddStaticShape(space, self.shape)
        self.line = draw.rectangle(
            start[0], start[1],
            end[0], end[1]+radius
            ,color)

    def render(self):
        self.line.draw(GL_QUADS)

class Box:
    def __init__(self, space, mass, width, height, pos, color=(255,0,0,255)):
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
        avg = (ps[0][0]+ps[1][0])/2
        points = []
        [ points.extend([p[0]-avg+avg%800, p[1]]) for p in ps ]
        pyglet.graphics.draw(4, GL_QUADS,
            ("v2f", points),
            ("c4B", list(self.color)*4)
            )


    def __str__(self):
        return "<box %s>"%self.body.contents.p


class Pendulum(reply.World):

    def __init__(self, filename):
        alpha = 1
        gamma = 0.6
        alpha_decay = 0.99
        min_alpha = 0.05
        epsilon = 0.9
        epsilon_decay = 0.99
        min_epsilon = 0.05
        self.filename = filename
        try:
            pend = cPickle.load(open(filename))
        except IOError:
            e = reply.encoder.DistanceEncoder(
                self.get_state_space(), self.get_action_space()
                )
            pend = reply.RL(
                    reply.learner.QLearner(alpha, gamma, alpha_decay, min_alpha),
                    reply.storage.DebugTableStorage(e),
                    e,
                    reply.selector.EGreedySelector(epsilon, epsilon_decay, min_epsilon)
                )
        self.rl = pend
        clock.set_fps_limit(30)
        self.win = window.Window(width=800, height=600)
        self.figure = None

        cp.cpInitChipmunk()

        #PID stuff
        self.integral = 0
        self.last_error = 0
        self.ki = 0.5
        self.kd = -0.5
        self.kp = 1.0



    def get_action_space(self):
        return [reply.dimension(-1,1,7)]

    def get_state_space(self):
        return [
            # ball relative x position
            reply.dimension(-84,84,6),
            # base velocity
            reply.dimension(-100,100,4),
            # ball x velocity
            reply.dimension(-100,100,2),
            # ball y velocity
            reply.dimension(-100,100,6),
        ]

    def get_reward(self, s ):
        died = 0
        if self.ball.body.contents.p.y < 100:
            return -1
        else:
            return 0
        if self.ball.body.contents.p.y < 100:
            died = -1000
        return (self.ball.body.contents.p.y-100 + died)

    def is_final(self, state):
        if self.ball.body.contents.p.y < 100:
            return True
        return False

    def get_state(self):
        return [
            self.ball.body.contents.p.x-self.base.body.contents.p.x,
            self.base.body.contents.v.x,
            self.ball.body.contents.v.x,
            self.ball.body.contents.v.y,
            ]

    def do_action(self, solver, action):
        # returns new state
        #print "DO:", action
        action = action[0]
        cp.cpBodyApplyForce(self.base.body, vec2d(action*10000,0), vec2d(0,0))
        cp.cpSpaceStep(self.space, dt+random.random()*1.0/100.0)
        cp.cpBodyResetForces(self.base.body)

        clock.tick()
        self.win.dispatch_events()

        self.win.clear()

        if self.figure: self.figure.blit(0,0)

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
        if self.rl.episodes%100==0:
            self.save()
        if False: #self.rl.episodes%10==0:
            self.integral = 0
            self.last_error = 0
            f = matplotlib.pyplot.figure()
            matplotlib.pyplot.matshow(self.rl.storage.debug_state, fignum=f.number,
                                      aspect=0.05)
            f.savefig("debug.png")
            self.figure = pyglet.image.load("debug.png")
        return self.get_state()

    def save(self):
        cPickle.dump(self.rl, open(self.filename, "w"))

    def cleanUp(self):
        cp.cpSpaceFree( self.space )

    def learn(self, iterations, max_steps=10000):
        try:
            for episode in xrange(iterations):
                self.rl.new_episode(self)
                for step in range(max_steps):
                    if not self.rl.step(self):
                        break
                print ('Espisode: ',episode,'  Steps:',step,
                        '  Reward:',self.rl.total_reward,
                        'alpha', self.rl.learner.alpha,
                        'epsilon', self.rl.selector.epsilon,
                        "avg_hist", self.rl.storage.median_hits,
                        "count_hits", self.rl.storage.count_hits)
        finally:
            self.save()

    def run_pid(self):
        while True:
            ended = False
            self.get_initial_state()
            while not ended:
                state = self.get_state()
                m = self.get_pid( state )
                self.do_action(None, [m])
                ended = self.is_final( state )
            self.cleanUp()

    def get_pid(self, state):
        e = self.pid_error(state)

        d = 1./5
        self.integral = self.integral*(1-d) + e*d
        derivative = e - self.last_error
        proportional = e
        self.last_error = e
        f = self.ki * self.integral + self.kd * derivative + self.kp * proportional
        print e, f, self.integral
        print self.get_state()
        return f

    def pid_error(self, state):
        sz = 100
        if abs( state[1]/sz ) > 1:
            return -1
        return -( math.acos( state[1]/sz ) - math.pi/2)


def run(maxepisodes):
    p = Pendulum("pendulum.pkl")
    p.learn(maxepisodes)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        p = Pendulum("foo")
        p.run_pid()
    else:
        run(1000000)


