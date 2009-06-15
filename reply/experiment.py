
class Experiment(object):

    def __init__(self):
        pass

    #
    # Standard API
    #

    def init(self):
        self.episodes = 1
        self.steps = 1
        self._init()
        self.initialized = True

    def start(self):
        self._start()
        self.started = True

    def step(self):
        result = self._step()
        return result

    def cleanup(self):
        self._cleanup()
        self.started = False
        self.initialized = False

    #
    # Overridable Methods
    #

    def _init(self):
        pass

    def _start(self):
        pass

    def _step(self):
        pass

    def _cleanup(self):
        pass
