from rlglue.environment.Environment import Environment
from rlglue.environment import EnvironmentLoader as EnvironmentLoader

__all__ = ["start_environment"]

class RlGlueProxyEnvironment(Environment):
    def __init__(self, reply_env):
        Environment.__init__(self)
        self.reply_env = reply_env
        
    def env_init(self):
        return self.reply_env.env_init(self)
    
    def env_start(self):
        return self.reply_env.env_start(self)
    
    def env_step(self, action):
        return self.reply_env.env_step(self, action)
    
    def env_cleanup(self):
        return self.reply_env.env_cleanup(self)
    
    def env_message(self, message):
        return self.reply_env.env_message(self, message)

            

def start_environment(env):
    rl_env = RlGlueProxyEnvironment(env)
    EnvironmentLoader.loadEnvironment(rl_env)
    