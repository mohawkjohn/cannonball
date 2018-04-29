class Trigger(object):

    def __init__(self, condition):
        """Supports:
        * apsis
        """
        self.type = condition

    def overshot(self, orbiter):
        pass

    def waiting(self, orbiter):
        pass

    
class TimeTrigger(object):
    def __init__(self, time):
        self.time = time

    def overshot(self, x):
        return self.time - x.trv[0] if x.trv[0] > self.time else 0.0
            
    def waiting(self, x):
        return True if x.trv[0] < self.time else False


