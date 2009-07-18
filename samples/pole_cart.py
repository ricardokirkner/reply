# This code is so you can run the samples without installing the package
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
#

import itertools
import math
import random

from reply.agent import LearningAgent
from reply.datatypes import Double, Integer, Model, Space
from reply.encoder import BucketEncoder, CompoundSpaceEncoder
from reply.environment import Environment
from reply.experiment import Experiment
from reply.learner import QLearner
from reply.policy import EGreedyPolicy
from reply.storage import TableStorage


GRAVITY = 9.8
MASSCART = 1.0
MASSPOLE = 0.1
TOTAL_MASS = (MASSPOLE + MASSCART)
LENGTH = 0.5 # actually half the pole's length
POLEMASS_LENGTH = (MASSPOLE * LENGTH)
FORCE_MAG = 10.0
TAU = 0.02 # dt
FOURTHIRDS = 1.3333333333333

# Common model
observations = Space(dict(
        velocity=Double(-1, 1),
        angle=Double(-90, 90),
        angle_velocity=Double(-1, 1)))
actions = Space({'force': Double(-1, 1)})
PoleCartModel = Model(observations, actions)


class PoleCartStateEncoder(CompoundSpaceEncoder):
    def __init__(self, space):
        encoders = {'velocity': BucketEncoder(observations.velocity, 10),
                    'angle': BucketEncoder(observations.angle, 36),
                    'angle_velocity': BucketEncoder(observations.angle_velocity, 10)}
        super(PoleCartStateEncoder, self).__init__(space, encoders)


class PoleCartActionEncoder(CompoundSpaceEncoder):
    def __init__(self, space):
        encoders = {'force': BucketEncoder(actions.force, 10)}
        super(PoleCartActionEncoder, self).__init__(space, encoders)


class PoleCartAgent(LearningAgent):
    model = PoleCartModel
    state_encoder_class = PoleCartStateEncoder
    action_encoder_class = PoleCartActionEncoder
    storage_class = TableStorage
    policy_class = EGreedyPolicy
    learner_class = QLearner

    learning_rate = 1
    learning_rate_decay = 0.999
    learning_rate_min = 0.001
    random_action_rate = 1



class PoleCartEnvironment(Environment):
    problem_type = "episodic"
    discount_factor = 1.0
    rewards = Integer(-1, 1)
    model = PoleCartModel

    def init(self):
        self.x = 0
        self.x_dot = 0
        self.theta = 0
        self.theta_dot = 0
        return super(PoleCartEnvironment, self).init()

    def start(self):
        return self.get_state()

    def get_state(self):
        return dict(velocity=self.x_dot, angle=math.degrees(self.theta),
                    angle_velocity=self.theta_dot)

    def step(self, action):
        force = action['force']

        costheta = math.cos(self.theta)
        sintheta = math.sin(self.theta)

        temp = ((force + POLEMASS_LENGTH * self.theta_dot * self.theta_dot * sintheta)
                             / TOTAL_MASS)

        thetaacc = ((GRAVITY * sintheta - costheta* temp)
                   / (LENGTH * (FOURTHIRDS - MASSPOLE * costheta * costheta
                                                  / TOTAL_MASS)))

        xacc  = temp - POLEMASS_LENGTH * thetaacc* costheta / TOTAL_MASS

        # Update the four state variables, using Euler's method.

        self.x  += TAU * self.x_dot
        self.x_dot += TAU * xacc
        self.theta += TAU * self.theta_dot
        self.theta_dot += TAU * thetaacc

        while self.theta > math.pi:
            self.theta -= 2.0 * math.pi

        while self.theta < -math.pi:
            self.theta += 2.0 * math.pi


        r = 0
        terminal = False
        if abs(math.degrees(self.theta)) > 80:
            terminal = True
            r = -1
        rot = self.get_state()
        rot.update(dict(reward=r, terminal=terminal))
        import pdb; pdb.set_trace()
        return rot


class PoleCartExperiment(Experiment):
    model = PoleCartModel


def run_simulation():
    # INTIALISATION
    import pygame, math, sys
    from pygame import draw
    from pygame.locals import *
    x_size, y_size = 640, 480
    screen = pygame.display.set_mode((x_size, y_size))
    clock = pygame.time.Clock()
    k_up = k_down = k_left = k_right = 0
    position = x_size/2, y_size-100
    height = -100
    top_position = position[0], position[1]+height
    while 1:
        # USER INPUT
        clock.tick(30)
        for event in pygame.event.get():
            if not hasattr(event, 'key'): continue
            down = event.type == KEYDOWN     # key down or up?
            if event.key == K_RIGHT: k_right = down * 5
            elif event.key == K_LEFT: k_left = down * 5
            elif event.key == K_UP: k_up = down * 2
            elif event.key == K_DOWN: k_down = down * 2
            elif event.key == K_ESCAPE: sys.exit(0)     # quit the game

        # RENDERING
        screen.fill((255,255,255))
        base = pygame.Rect(0,0,75,50)
        base.center = position
        draw.line(screen, (0,0,0), position, top_position, 3)
        draw.rect(screen, (0,0,0), base)
        draw.circle(screen, (0,0,0), top_position, 30)
        floor = pygame.Rect(0,0,x_size, 75)
        floor.bottom = y_size
        draw.rect(screen, (100,100,100), floor)
        pygame.display.flip()

if __name__=="__main__":
    run_simulation()
    from reply.runner import Runner
    r = Runner(PoleCartAgent(), PoleCartEnvironment(), PoleCartExperiment())
    r.run()
