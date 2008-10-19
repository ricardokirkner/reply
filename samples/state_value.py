# This code is so you can run the samples without installing the package
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
#

import reply
import numpy
from numpy import array
import random


class StateValue(reply.World):
    def __init__(self, rl):
        " ps == win probability array for each state (implies number of states)"
        super(StateValue, self).__init__(rl)
        self.ps = getattr(rl, "state_probability_array")

    def is_final(self):
        return True

    def do_action(self, action):
        selection = action[0]
        if random.random() < self.ps[int(self.state)]:
            self.won = True


    def new_episode(self):
        self.won = 0
        self.state = random.randint(1,len(self.ps)-1)

    def end_episode(self):
        err = 0
        for i, s in enumerate(self.rl.get_state_space()[0]):
            e = self.rl.encoder.encode_state( dict(state=s) )
            v = self.rl.storage.get_max_value( e )
            err += abs(self.ps[i]-v)
        self.rl.current_episode.error = err

    def get_state(self):
        return dict(
                state=self.state
            )

class StateValueAgent(reply.LearningAgent):
    state_probability_array = [ p/10.0 for p in range(10) ]

    learning_rate = 1
    learning_rate_decay = 0.99
    learning_rate_min = 0.001

    random_action_rate = 1

    world_class = StateValue
    learner_class = reply.learner.QLearner
    selector_class = reply.selector.EGreedySelector
    storage_class = reply.storage.TableStorage
    encoder_class = reply.encoder.DistanceEncoder

    def get_action_space(self):
        return [ reply.Dimension("pass", 1,1) ]

    def get_state_space(self):
        return [ reply.Dimension("state", 0,len(self.world.ps)-1) ]

    def get_reward(self):
        if self.world.won:
            return 1
        return 0

if __name__ == "__main__":
    reply.Experiment(StateValueAgent()).run()
    ps = [ p/10.0 for p in range(10) ]
    g = StateValue(ps)
    e = reply.encoder.DistanceEncoder(g.get_state_space(), g.get_action_space())
    r = reply.RL(
            reply.learner.QLearner(0.001, 0.01,1),
            reply.storage.TableStorage(e),
            e,
            reply.selector.EGreedySelector(0.1, 1)
        )

    for episode in range(10000):
        total_reward,steps  = r.run(g)
        err = 0
        for i, s in enumerate(g.get_state_space()[0]):
            e = r.encoder.encode_state( [ s ] )
            v = r.storage.get_max_value(
                    r.encoder.encode_state( [ s ] )
                )
            err += abs(ps[i]-v)

        print 'Espisode: ',episode,'  Steps:',steps,'  Reward:',total_reward,' error:', err
