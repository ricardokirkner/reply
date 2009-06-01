
class Experiment(object):

    def __init__(self):
        pass

    #
    # Standard API
    #

    def init(self):
        self._init()
        self.initialized = True

    def start(self):
        self._start()
        self.started = True

    def step(self):
        self._step()

    def run(self):
        pass

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
