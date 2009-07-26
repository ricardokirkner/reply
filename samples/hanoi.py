import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import itertools
import math
import simplejson

import pygame
from pygame import draw
from pygame.locals import K_SPACE, K_ESCAPE

from reply.agent import LearningAgent
from reply.datatypes import Integer, Model, Space
from reply.environment import Environment
from reply.experiment import Experiment
from reply.learner import QLearner
from reply.policy import EGreedyPolicy
from reply.storage import TableStorage

DEBUG, INFO, NONE = range(3)
LOG_LEVEL = NONE


class Peg(object):
    def __init__(self, number):
        self.number = number
        self.discs = []

    @property
    def is_valid(self):
        return sorted(self.discs, reverse=True) == self.discs

    @property
    def num_discs(self):
        return len(self.discs)

    def fill(self, num_discs):
        self.discs = list(reversed(range(num_discs)))

    def empty(self):
        self.discs = []

    def pop(self):
        return self.discs.pop()

    def push(self, disc):
        self.discs.append(disc)

    def __repr__(self):
        return '<%s: number: %d discs: %s>' % (
            self.__class__.__name__, self.number, str(self.discs))

    def __contains__(self, disc):
        return disc in self.discs


def build_model(num_pegs=3, num_discs=3):
    discs = ['disc_%d' % disc for disc in xrange(num_discs)]
    spec = {}
    for disc in discs:
        spec[disc] = Integer(0, num_pegs-1)

    observations = Space(spec)
    actions = Space({'from_peg': Integer(0, num_pegs-1),
                     'to_peg': Integer(0, num_pegs-1)},
                     valid=is_valid_action)
    model = Model(observations, actions)
    return model

def is_valid_action(action):
    return action['from_peg'] != action['to_peg']


class HanoiAgent(LearningAgent):
    model = build_model()
    storage_class = TableStorage
    policy_class = EGreedyPolicy
    learner_class = QLearner

    learning_rate = 0.9
    learning_rate_decay = 0.999
    learning_rate_min = 0.0001
    value_discount = 0.8
    random_action_rate = 0.9
    random_action_rate_decay = 0.995


class HanoiEnvironment(Environment):
    problem_type = "episodic"
    discount_factor = 0.8
    rewards = Integer(-1, 1)
    initial_peg = 0
    model = build_model()
    last_action = None

    def init(self):
        self.pegs = [Peg(i) for i in xrange(self.num_pegs)]
        return super(HanoiEnvironment, self).init()

    def start(self):
        # reset all pegs
        [peg.empty() for peg in self.pegs]
        # initialize
        self.pegs[self.initial_peg].fill(self.num_discs)

        observation = self.get_state()
        return observation

    def step(self, action):
        from_peg_num = action['from_peg']
        to_peg_num = action['to_peg']
        if LOG_LEVEL <= DEBUG:
            print 'moving from peg %d to peg %d' % (from_peg_num, to_peg_num)
        from_peg = self.pegs[int(from_peg_num)]
        to_peg = self.pegs[int(to_peg_num)]

        if not from_peg.discs:
            rot = self.get_state()
            rot['reward'] = -1
            rot['terminal'] = True
            if LOG_LEVEL <= DEBUG:
                print "move invalid: empty"
                print "rot:", rot
            return rot

        if to_peg.discs and from_peg.discs[-1] > to_peg.discs[-1]:
            rot = self.get_state()
            rot['reward'] = -1
            rot['terminal'] = True
            if LOG_LEVEL <= DEBUG:
                print "move invalid: too big"
                print "rot:", rot
            return rot

        if self.last_action is not None:
            if to_peg_num == self.last_action['from_peg'] and \
               from_peg_num == self.last_action['to_peg']:
                rot = self.get_state()
                rot['reward'] = -1
                rot['terminal'] = True
                if LOG_LEVEL <= DEBUG:
                    print 'move invalid: backtracking'
                    print 'rot:', rot
                return rot

        disc = from_peg.pop()
        to_peg.push(disc)
        if LOG_LEVEL <= INFO:
            print "\t".join( [ str(peg.discs) for peg in self.pegs ] )

        self.last_action = action
        rot = self.get_state()
        rot['reward'] = self.get_reward()
        rot['terminal'] = self.is_final()
        #if rot['terminal']: print 'FINAL...'
        return rot

    def get_reward(self):
        reward = 0
        for peg in self.pegs:
            if not peg.is_valid:
                # invalid disc layout
                if LOG_LEVEL <= DEBUG:
                    print 'peg is invalid', peg.discs
                reward = -1
                break
            elif self.discs_moved(peg):
                # all discs moved to another peg
                if LOG_LEVEL <= DEBUG:
                    print 'all discs moved to peg %d' % peg.number, peg.discs
                reward = 1
                break
        return reward

    def is_final(self):
        is_final = False
        for peg in self.pegs:
            is_final |= self.discs_moved(peg)
        return is_final

    def get_state(self):
        state = {}
        for disc in range(self.num_discs):
            state["disc_%d" % disc] = self.find_peg(disc)
        return state

    def discs_moved(self, peg):
        return peg.number != self.initial_peg and \
               peg.num_discs == self.num_discs

    def find_peg(self, disc):
        for i,peg in enumerate(self.pegs):
            if disc in peg:
                return i
        raise Exception("disc not found")

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, str(self.pegs))

    def on_get_num_pegs(self):
        return self.num_pegs

    def on_get_num_discs(self):
        return self.num_discs


class HanoiExperiment(Experiment):
    max_episodes = 1000

    def setup(self):
        num_pegs = self.env_call('get_num_pegs')
        scale = num_pegs / 3.0

        x_size, y_size = 640 * scale, 480 * scale
        self.screen = pygame.display.set_mode((x_size, y_size))
        self.clock = pygame.time.Clock()

        k_up = k_down = k_left = k_right = 0
        position = x_size/2, y_size-100
        height = -100
        top_position = position[0], position[1]+height

        self.x_size, self.y_size = x_size, y_size
        self.slow = False
        self.step_average = 0

    def handle_events(self):
        for event in pygame.event.get():
            if not hasattr(event, 'key'):
                continue
            elif event.key == K_SPACE:
                self.slow = not self.slow
            elif event.key == K_ESCAPE:
                # quit
                sys.exit(0)

    def draw(self, observation):
        if self.slow:
            self.clock.tick(5)
            self.screen.fill((240,240,255))
        else:
            self.screen.fill((255,255,255))

        num_pegs = self.env_call('get_num_pegs')
        num_discs = self.env_call('get_num_discs')

        pegs = []
        for peg in range(num_pegs):
            pegs.append([])

        for disc in range(num_discs-1,-1,-1):
            peg = observation['disc_%d' % disc]
            pegs[peg].append(disc)

        for p in range(num_pegs):
            # draw the peg
            px = 100 + p*200
            peg = pygame.Rect(px, 275, 20, 200)
            peg.bottom = self.y_size - 75
            draw.rect(self.screen, (100, 100, 100), peg)

            # draw the discs
            for i,d in enumerate(pegs[p]):
                w, h = 150 - (num_discs-1-d)*20, 20
                d_left = px - (w-20)/2
                d_top = 75+h*i
                disc = pygame.Rect(d_left, d_top, w, h)
                disc.bottom = self.y_size - d_top
                draw.rect(self.screen, (200,0,200), disc)

        floor = pygame.Rect(0,0,self.x_size,75)
        floor.bottom = self.y_size
        draw.rect(self.screen, (100,100,100), floor)

        pygame.display.flip()

    def run(self):
        # setup screen
        self.setup()

        # initialize experiment
        self.init()

        for episode in range(self.max_episodes):
            self.start()
            steps = 0
            terminal = False
            while steps < self.max_steps and not terminal:
                roat = self.step()
                terminal = roat['terminal']

                self.handle_events()
                self.draw(roat)

                steps += 1

            alpha = 0.05
            self.step_average = alpha*steps + (1-alpha)*self.step_average
            print "Episode:", episode, "epsilon", self.glue_experiment.agent.policy.random_action_rate, "Steps:", steps, "avg:", self.step_average
        self.cleanup()


if __name__ == "__main__":
    from reply.runner import Run, Runner, register_command, unregister_command
    class HanoiRun(Run):
        def register(self, parser):
            super(HanoiRun, self).register(parser)
            parser.add_argument("--num-discs", type=int, dest="num_discs",
                                help="the number of discs in the experiment")
            parser.add_argument("--num-pegs", type=int, dest="num_pegs",
                                help="the number of pegs in the experiment")
            parser.add_argument("--initial-peg", type=int, dest="initial_peg",
                                help="the initial peg number")
            #parser.add_argument("--log-level", type=int, dest="log_level",
            #                    help="verbosity (0|1|2)")

        def run(self, agent, env, experiment, args=None):
            self.agent = agent
            self.env = env
            self.experiment = experiment

            model_updated = False

            if args is not None:
                if args.max_episodes is not None:
                    experiment.max_episodes = args.max_episodes
                if args.max_steps is not None:
                    experiment.max_steps = args.max_steps
                if args.num_discs is not None:
                    env.num_discs = args.num_discs
                    model_updated = True
                if args.num_pegs is not None:
                    env.num_pegs = args.num_pegs
                    model_updated = True
                if args.initial_peg is not None:
                    env.initial_peg = args.initial_peg
                #if args.log_level is not None:
                #    LOG_LEVEL = args.log_level

            if model_updated:
                model = build_model(env.num_pegs, env.num_discs)
                env.model = model
                agent.model = model

            self.experiment.set_glue_experiment(self)
            self.experiment.run()


    r = Runner(HanoiAgent(), HanoiEnvironment(), HanoiExperiment())
    # replace default run command with a custom one
    unregister_command(Run)
    register_command(HanoiRun)
    # run the experiment
    r.run()
