# This code is so you can run the samples without installing the package
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
#

import itertools
import math
import random

import pygame, math, sys
from pygame import draw
from pygame.locals import *

from reply.agent import LearningAgent
from reply.datatypes import Double, Integer, Model, Space
from reply.environment import Environment
from reply.experiment import Experiment
from reply.learner import QLearner
from reply.policy import EGreedyPolicy
from reply.storage import TableStorage
from reply.mapping import TileMapping


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
        velocity=Double(-10, 10),
        angle=Double(-math.pi/2, math.pi/2),
        angle_velocity=Double(-10, 10),
        position=Double(-1000,1000)))

actions = Space({'force': Double(-1, 1)})
PoleCartModel = Model(observations, actions)


class PoleCartAgent(LearningAgent):
    model = PoleCartModel
    storage_class = TableStorage
    storage_observation_buckets = dict(position=1, velocity=10, angle=20, angle_velocity=10)
    storage_action_buckets = dict(force=20)
    policy_class = EGreedyPolicy
    learner_class = QLearner

    learning_rate = 0.3
    learning_rate_decay = 0.999
    learning_rate_min = 0.005
    random_action_rate = 1
    random_action_rate_decay = 0.99
    random_action_rate_min = 0.01

    def build_storage(self):
        self.storage = self.storage_class(self,
                TileMapping(observations, dict(velocity=10, position=1, angle=20, angle_velocity=10)),
                TileMapping(actions, dict(force=20)))

def cap(n, m, M):
    return max(m, min(M, n))

class PoleCartEnvironment(Environment):
    problem_type = "episodic"
    discount_factor = 1.0
    rewards = Integer(-1, 1)
    model = PoleCartModel

    def init(self):
        return super(PoleCartEnvironment, self).init()

    def start(self):
        self.x = 0
        self.x_dot = 0
        self.theta = 0
        self.theta_dot = 0
        return self.get_state()

    def get_state(self):
        return dict(position=cap(self.x,
                                 observations.position.min,
                                 observations.position.max),
                    velocity=cap(self.x_dot,
                                 observations.velocity.min,
                                 observations.velocity.max),
                    angle=cap(self.theta,
                                       observations.angle.min,
                                       observations.angle.max),
                    angle_velocity=cap(self.theta_dot,
                                       observations.angle_velocity.min,
                                       observations.angle_velocity.max))

    def step(self, action):
        force = (action['force']*50) * (1 + random.random()/10 - 0.05)
        #print "FORCE", action['force']*50

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
        return rot


class PoleCartExperiment(Experiment):
    model = PoleCartModel
    max_steps = 10000
    def run(self):
        x_size, y_size = 640, 480
        screen = pygame.display.set_mode((x_size, y_size))
        clock = pygame.time.Clock()
        k_up = k_down = k_left = k_right = 0
        position = x_size/2, y_size-100
        height = -100
        top_position = position[0], position[1]+height
        self.init()
        slow = False
        step_average = 0
        for episode in range(self.max_episodes):
            self.start()
            steps = 0
            terminal = False
            while steps < self.max_steps and not terminal:
                roat = self.step()
                terminal = roat["terminal"]
                for event in pygame.event.get():
                    if not hasattr(event, 'key'): continue
                    elif event.key == K_SPACE: slow = not slow
                    elif event.key == K_ESCAPE: sys.exit(0)     # quit

                # RENDERING
                tetha = roat["angle"]
                length = 150
                position = (x_size/2 + roat["position"]*length)%x_size,y_size-100
                top_position = position[0] + math.sin(tetha)*length, \
                            position[1] - math.cos(tetha)*length
                if slow:
                    clock.tick(30)
                    screen.fill((240,240,255))
                else:
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
                steps += 1
            alpha = 0.05
            step_average = alpha*steps + (1-alpha)*step_average
            print "Episode:", episode, "epsilon", self.glue_experiment.agent.policy.random_action_rate, "Steps:", steps, "avg:", step_average
        self.cleanup()

def run_simulation():
    # INTIALISATION


    while 1:
        # USER INPUT
        clock.tick(30)


if __name__=="__main__":
    #run_simulation()
    from reply.runner import Runner
    r = Runner(PoleCartAgent(), PoleCartEnvironment(), PoleCartExperiment())
    r.run()
