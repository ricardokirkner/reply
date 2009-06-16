# This code is so you can run the samples without installing the package
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
#

import random

from reply.agent import Agent
from reply.datatypes import Integer
from reply.encoder import SpaceEncoder, StateActionEncoder
from reply.environment import Environment
from reply.experiment import Experiment
from reply.learner import QLearner
from reply.policy import EGreedyPolicy
from reply.storage import TableStorage


class ActionValueAgent(Agent):
    def _init(self, task_spec):
        learning_rate = 1
        learning_rate_decay = 0.999
        learning_rate_min = 0.00001
        random_action_rate = 1

        state_encoder = SpaceEncoder(self._observation_space)
        action_encoder = SpaceEncoder(self._action_space)
        encoder = StateActionEncoder(state_encoder, action_encoder)
        storage = TableStorage((1, 10), encoder)
        policy = EGreedyPolicy(storage, random_action_rate)
        self.learner = QLearner(policy, learning_rate, learning_rate_decay,
                                learning_rate_min)

        self.last_observation = None
        self.last_action = None

    def _start(self, observation):
        self.learner.new_episode()
        action = self.learner.policy.select_action(observation)
        self.last_observation = observation
        self.last_action = action
        print "chosen action", action
        return action

    def _step(self, reward, observation):
        self.learner.update(self.last_observation, self.last_action, reward,
                            observation)
        action = self.learner.policy.select_action(observation)
        self.last_observation = observation
        self.last_action = action
        return action

    def _end(self, reward):
        self.learner.update(self.last_observation, self.last_action, reward,
                            None)
        print "reward", reward
        print self.learner.policy.storage.data[0]
        print "learning rate", self.learner.learning_rate

class ActionValueEnvironment(Environment):
    actions_spec = {'choice': Integer(0, 9)}
    observations_spec = {'state': Integer(1, 1)}
    problem_type = "episodic"
    discount_factor = 1.0
    rewards = Integer(-1, 1)

    def on_set_num_action(self, n):
        self.set_action_space(choice=Integer(0, n-1))

    def _init(self):
        print "env init"
        maxval = self._action_space["choice"].max + 1
        self.ps = [ (p+1)/float(maxval) for p in range(maxval) ]

    def _start(self):
        print "env_start"
        return dict(state=0)

    def _step(self, action):
        if random.random() < self.ps[action['choice']]:
            r = 1
        else:
            r = 0
        rot = dict(state=0, reward=r, terminal=True)
        return rot

    def _end(self, reward):
        pass


class ActionValueExperiment(Experiment):
    pass

if __name__=="__main__":
    import sys
    def usage():
        print "%s [agent|environment|experiment|runner]" % sys.argv[0]

    if len(sys.argv) < 2:
        usage()
        exit(0)

    role = sys.argv[1]
    if role == 'agent':
        from reply.glue import start_agent
        start_agent(ActionValueAgent())
    elif role == 'environment':
        from reply.glue import start_environment
        start_environment(ActionValueEnvironment())
    elif role == 'experiment':
        from reply.glue import start_experiment
        start_experiment(ActionValueExperiment())
    elif role == 'runner':
        from multiprocessing import Process
        from reply.glue import start_agent
        from reply.glue import start_environment
        from reply.glue import start_experiment

        class ActionValueRunner():
            def run(self):
                agent = Process(target=start_agent,
                                args=(ActionValueAgent(),))
                environment = Process(target=start_environment,
                                      args=(ActionValueEnvironment(),))
                experiment = Process(target=start_experiment,
                                     args=(ActionValueExperiment(),))

                agent.start()
                environment.start()
                experiment.start()

                agent.join()
                environment.join()
                experiment.join()

        runner = ActionValueRunner()
        runner.run()
    elif role == 'reply_runner':
        from reply.reply_glue import ReplyRunner

        runner = ReplyRunner(ActionValueAgent(), ActionValueEnvironment(),
                             ActionValueExperiment())
        runner.run()
    else:
        usage()
        exit(0)
