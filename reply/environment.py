import simplejson

from reply.types import Space

class Environment(object):
    problem_type = "episodic"
    discount_factor = 1.0
    rewards = 0,1
    
    def __init__(self):
        self.initialized = False
        self.set_action_space(**self.action_space)
        self.set_observation_space(**self.observation_space)
        
    def set_observation_space(self, **kwargs):
        if self.initialized:
            raise Exception("Can't change observation space after init")
        self._observation_space = Space(kwargs)
        
    def set_action_space(self, **kwargs):
        if self.initialized:
            raise Exception("Can't change action space after init")
        self._action_space = Space(kwargs)
        
    def env_init(self):
        self.init()
        self.initialized = True
        return self.get_task_spec()
    
    
    def get_task_spec(self):
        return ("VERSION RL-Glue-3.0 " +
                "PROBLEMTYPE %s " % self.problem_type + 
                "DISCOUNTFACTOR %s " % self.discount_factor +
                "OBSERVATIONS %s " % self._observation_space +
                "ACTIONS %s " % self._action_space +
                "REWARDS (%s %s) " % self.rewards +
                "EXTRA %s" % self.__doc__)
    
    def env_start(self):
        pass
    
    def env_step(self, action):
        pass
    
    def env_cleanup(self):
        pass
    
    def env_message(self, message):
        message = simplejson.parse(message)
        fname = message['function_name']
        f = getattr(self, "on_"+fname, None)
        if f is not None:
            f(*message['args'], **f['kwargs'])
    
    def init(self):
        pass