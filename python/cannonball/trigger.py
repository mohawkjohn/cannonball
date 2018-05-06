class Trigger(object):
    
    def is_exact(self):
        return False

    
class ApsisTrigger(Trigger):
    """Stop integrating when an apsis is reached."""

    def __init__(self, mode = 'periapsis'):
        self.mode = mode
    
    def is_exact(self):
        return False
    
    def __lt__(self, x):
        if self.mode == 'periapsis':
            return x.radial_velocity() > 0.0
        else:
            return x.radial_velocity() < 0.0

    def __eq__(self, x):
        if x.radial_velocity() != 0.0:
            return False
        if self.mode == 'periapsis':
            return x.radial_acceleration() > 0.0
        else:
            return x.radial_acceleration() < 0.0

    def __gt__(self, x):
        if self.mode == 'periapsis':
            return x.radial_velocity() < 0.0
        else:
            return x.radial_velocity() > 0.0        

    def guess_remaining(self, x):
        """How much time remains before we reach apsis?

        TODO: Produce an exact solution.
        """
        return x.estimate_time_to_next_apsis()

    
class TimeTrigger(Trigger):
    """Stop integrating at a specific time."""
    
    def __init__(self, time):
        self.time  = time

    def is_exact(self):
        return True

    def __lt__(self, x):
        """If this trigger is less than some state, it means the state has
        passed the trigger."""
        return self.time < x.trv[0]

    def __gt__(self, x):
        return self.time > x.trv[0]

    def guess_remaining(self, x):
        """Attempt to guess the remaining propagation time.
        For TimeTrigger, the guess is exact (indicated by is_exact()).
        For other triggers, the guess is guaranteed to be an underestimate
        (so we'll never overshoot).
        """
        return self.time - x.trv[0]
