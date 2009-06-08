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
from reply.policy import EGreedyPolicy
from reply.storage import TableStorage


class ActionValueAgent(Agent):
    def _init(self, task_spec):
        learning_rate = 1
        learning_rate_decay = 0.99
        learning_rate_min = 0.001
        random_action_rate = 1

        #self.learner = reply.learner.QLearner(None)
        state_encoder = SpaceEncoder(self._observation_space)
        action_encoder = SpaceEncoder(self._action_space)
        encoder = StateActionEncoder(state_encoder, action_encoder)
        storage = TableStorage((1, 10), encoder)
        self.policy = EGreedyPolicy(storage, random_action_rate)
        pass

    def _start(self, observation):
        #encoded_observation = observation['state']
        import pdb; pdb.set_trace()
        #encoded_observation = self.encoder['observation'].encode(observation)
        #encoded_action = self.policy.select_action(encoded_observation)
        ##action = {'choice': encoded_action}
        #action = self.encoder['action'].decode(encoded_action)
        action = self.policy.select_action(observation)
        return action
        #choice = self._action_space['choice']
        #action = {'choice': random.choice(range(choice.min, choice.max))}
        #return action

    def _step(self, reward, observation):
        ##encoded_observation = observation['state']
        #encoded_observation = self.encoder['observation'].encode(observation)
        #encoded_action = self.policy.select_action(encoded_observation)
        ##action = {'choice': encoded_action}
        #action = self.encoder['action'].decode(encoded_action)
        action = self.policy.select_action(observation)
        return action
        #choice = self._action_space['choice']
        #action = {'choice': random.choice(range(choice.min, choice.max))}
        #return action


class ActionValueEnvironment(Environment):
    actions_spec = {'choice': Integer(0, 9)}
    observations_spec = {'state': Integer(0, 0)}
    problem_type = "episodic"
    discount_factor = 1.0
    rewards = Integer(-1, 1)

    def on_set_num_action(self, n):
        self.set_action_space(choice=Integer(0, n-1))

    def _init(self):
        maxval = self._action_space["choice"].max
        self.ps = [ p/float(maxval) for p in range(maxval+1) ]

    def _start(self):
        return dict(state=0)

    def _step(self, action):
        print str(action)
        if random.random() > self.ps[action['choice']]:
            r = 1
        else:
            r = 0
        rot = dict(state=0, reward=r, final=True)
        print rot
        return rot


class ActionValueExperiment(Experiment):
    def _init(self):
        pass

    def _start(self):
        pass

    def _step(self):
        pass

    def _cleanup(self):
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
    else:
        usage()
        exit(0)

