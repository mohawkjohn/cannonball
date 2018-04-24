import rk4

from log import Log

import gravity

def make_elements(r_periapsis  = None,
                  eccentricity = None,
                  inclination  = None,
                  mu           = None,
                  epoch_time   = None):
    elements = np.array([r_periapsis, eccentricity, inclination, 0.0, 0.0, 0.0, epoch_time, mu])
    return elements

class Orbiter(object):
    """Just an object that is affected by at least one body."""
    
    def __init__(self,
                 trv          = None,
                 elements     = None,
                 epoch_time   = 0.0,
                 grav_bodies  = []):
        if elements is not None:
            import spiceypy as spice
            trv = np.hstack((0.0, spice.conics(elements, epoch_time)))
        
        self.trv         = trv
        self.log         = Log('trv.log', mode = 'w')

        self.log.write(trv)

    def r_INERTIAL(self):
        return self.trv[1:4]

    def v_INERTIAL(self):
        return self.trv[4:7]

    def integrate(self, dt_requested, t_max, accuracy=1e-6, gradient = False, **step_kwargs):
        dt     = dt_requested
        dt_min = dt
        dt_max = dt

        while self.trv[0] < t_max:
            dt, trv, temp = rk4.adaptive_step(self.trv, dt, gravity.basic_gravity, accuracy, step_using_index = False, **step_kwargs)
            print "trv time is ", trv[0]
            dt_min = min(dt, dt_min)
            dt_max = max(dt, dt_min)

            if trv[0] > t_max: # Try again with a smaller step
                print "trying again with smaller step from ", self.trv[0], " to ", t_max
                trv, temp = rk4.step(self.trv, t_max - self.trv[0], gravity.basic_gravity, step_using_index = False, **step_kwargs)
                self.trv = trv
            else:
                self.trv = trv

            self.log.write(trv)

        return
