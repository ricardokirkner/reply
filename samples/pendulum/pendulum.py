# This code is so you can run the samples without installing the package
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
#

import random, cPickle
import math


import pymunk._chipmunk as cp
import pymunk.util as u
from pymunk import Vec2d as vec2d
import ctypes

from numpy import arange, array, size
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


class PendulumWorld(reply.World):

    def __init__(self, rl):
        super(PendulumWorld, self).__init__(rl)
        cp.cpInitChipmunk()

    def is_final(self):
        if self.ball.body.contents.p.y < 100:
            return True
        return False

    def get_state(self):
        return dict(
            ball_angle=self.ball_angle(),
            ball_angular_v=self.ball_angle() - self.last_angle,
            base_vx=self.base.body.contents.v.x
            )

    def do_action(self, action):
        action = action[0]
        self.save_angle()
        cp.cpBodyApplyForce(self.base.body, vec2d(action*15000,0), vec2d(0,0))
        cp.cpSpaceStep(self.space, dt)
        cp.cpBodyResetForces(self.base.body)

    def new_episode(self):
        # return first state
        self.space = space = cp.cpSpaceNew()
        space.contents.gravity = vec2d(0.0, -900.0)
        space.contents.damping = 0.01
        
        cp.cpSpaceResizeStaticHash(space, 50.0, 200)
        cp.cpSpaceResizeActiveHash(space, 50.0, 10)

        self.ball = ball = Ball(space, 5, 10, pos=(310, 180))
        self.floor = Wall(space, (-60000,40), (60000,40))
        self.base = base = Box(space, 5, 50, 20, (300,100))
        cp.cpResetShapeIdCounter()
        joint = cp.cpPinJointNew(
            ball.body, base.body, vec2d(0,0), vec2d(0,0)
            )
        cp.cpSpaceAddJoint(space, joint)
        self.save_angle()
        
    def ball_angle(self):
        x = self.ball.body.contents.p.x
        y = self.ball.body.contents.p.y
        cx = self.base.body.contents.p.x
        cy = self.base.body.contents.p.y

        x = x-cx
        y = y-cy
        if y == 0: return 0
        return math.atan( x/y )

    def save_angle(self):
        self.last_angle = self.ball_angle()
        
    def end_episode(self):
        #self.space.cpDestroy()
        pass

    def save(self):
        cPickle.dump(self.rl, open(self.filename, "w"))

    def cleanUp(self):
        cp.cpSpaceFree( self.space )


class PendulumAgent(reply.Agent):
    random_action_rate = 0.05

    world_class = PendulumWorld
    selector_class = reply.selector.EGreedySelector
    storage_class = reply.storage.TableStorage
    encoder_class = reply.encoder.DistanceEncoder

    learning_rate = 0.3
    learning_rate_decay = 0.999
    learning_rate_min = 0.005
    value_discount = 0.5

    random_action_rate = 0.99
    random_action_rate_decay = 0.99
    random_action_rate_min = 0.005

    learner_class = reply.learner.QLearner
    
    def get_action_space(self):
        return [reply.Dimension("push", -1,1,17)]

    def get_state_space(self):
        return [
            # ball angle
            reply.Dimension("ball_angle", -math.pi/2,math.pi/2,11),
            # ball angular velocity
            reply.Dimension("ball_angular_v", -math.pi/16,math.pi/16,11),
            # base velocity
            reply.Dimension("base_vx", -100,100,11),
        ]

    def get_reward(self):
        if self.world.ball.body.contents.p.y < 100:
            return -1000
        if abs(self.world.ball_angle()) < math.pi/16:
            return math.pi/16 - abs(self.world.ball_angle())
        return 0
    
    
if __name__ == "__main__":
    agent = PendulumAgent()
    experiment = reply.Experiment(agent)
    experiment.run()


