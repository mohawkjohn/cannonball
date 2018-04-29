import rk4

from log import Log

import gravity
import spiceypy as spice

import numpy as np
from scipy.linalg import norm

def make_elements(r_periapsis  = None,
                  eccentricity = None,
                  inclination  = None,
                  mu           = None,
                  epoch_time   = None):
    elements = np.array([r_periapsis, eccentricity, inclination, 0.0, 0.0, 0.0, epoch_time, mu])
    return elements


class Orbiter(object):
    """Just an object that is affected by at least one body."""

    class State(object):
        def __init__(self,
                     trv        = None,
                     elements   = None,
                     epoch_time = 0.0,
                     conic_mu   = 3.986004415e14):
            self.conic_mu    = conic_mu

            if elements is not None:
                # Note that SPICE uses km, km/s, and we use m, m/s
                trv = np.hstack((0.0, spice.conics(elements, epoch_time) * 1000.0))

            self.trv          = trv

        def copy(self):
            """ Make a copy of this object but without the logs."""
            return Orbiter.State(np.array(self.trv), conic_mu = self.conic_mu)
            
        def r_INERTIAL(self):
            return self.trv[1:4]

        def v_INERTIAL(self):
            return self.trv[4:7]

        def conic_elements(self):
            """Compute the conic elements of the orbiter at the current
            time. Uses SPICE.
            """
            time     = self.trv[0]
            rv_km    = self.trv[1:] / 1000.0 # convert to km, km/sec
            elements = spice.oscelt(rv_km, time, self.conic_mu * 1e-9)
            elements[0] *= 1000.0 # convert perifocal distance to meters

            return elements

        def r_periapsis(self):
            return self.conic_elements()[0]

        def r_apoapsis(self):
            # rp = a * (1 - e) => a = rp / (1 - e)
            # ra = a * (1 + e) => ra = rp * (1 + e) / (1 - e)
            elements = self.conic_elements()
            rp = elements[0]
            e  = elements[1]
            return rp * (1.0 + e) / (1.0 - e)

        def specific_energy(self):
            v2    = self.trv[4:7].dot(self.trv[4:7])
            rmag  = norm(self.trv[1:4])
            return v2 * 0.5 - self.conic_mu / rmag

        def ecc(self):
            return self.conic_elements()[1]

        def inc(self):
            return self.conic_elements()[2]

        def lnode(self):
            """Longitude of the ascending node"""
            return self.conic_elements()[3]

        def argp(self):
            """Argument of periapsis"""
            return self.conic_elements()[4]

        def mean_anomaly_at_epoch(self):
            return self.conic_elements()[5]

        def semimajor_axis(self):
            # FIXME: This is elliptical only; for hyperbola, should be
            # negative.
            elements = self.conic_elements()
            rp = elements[0]
            e  = elements[1]
            a  = rp / (1.0 - e)
            return rp / (1.0 - e)
        
        def adaptive_step(self, dt, **step_kwargs):
            dt, trv, temp = rk4.adaptive_step(self.trv, dt, flow = gravity.basic_gravity, mu = self.conic_mu, **step_kwargs)
            return Orbiter.State(trv = trv, conic_mu = self.conic_mu), dt

        def fixed_step(self, dt, **step_kwargs):
            trv, temp = rk4.step(self.trv, dt, flow = gravity.basic_gravity, mu = self.conic_mu, **step_kwargs)
            return Orbiter.State(trv = trv, conic_mu = self.conic_mu)
        
        
    
    def __init__(self,
                 use_logs     = {'inertial': 'inertial.log',
                                 'elements': 'elements.log'},
                 **state_kwargs):
        """ Constructor. Must provide either trv or elements/epoch_time.
        Generally expects everything to be in meters.

        Args:
            trv:        initial time, position, velocity vector
            elements:   orbital elements, e.g. from make_elements()
            epoch_time: epoch time of elements (default: 0.0)
            conic_mu:   gravitational constant for central body (e.g. earth)
                        for conic approximation (default is earth); should be
                        in m^3/s^2"""
        self.state = Orbiter.State(**state_kwargs)

        self.logs         = {}
        if use_logs['inertial']:
            self.logs['inertial'] = Log(use_logs['inertial'], mode = 'w')
            self.log_inertial()
        if use_logs['elements']:
            self.logs['elements'] = Log(use_logs['elements'], mode = 'w')
            self.log_elements()

    def integrate(self, dt_requested, t_max, accuracy = 1e-6, gradient = False, **step_kwargs):
        dt     = dt_requested
        dt_min = dt
        dt_max = dt

        while self.state.trv[0] < t_max:
            state, dt = self.state.adaptive_step(dt, accuracy = accuracy, **step_kwargs)
            dt_min = min(dt, dt_min)
            dt_max = max(dt, dt_min)

            if state.trv[0] > t_max: # Try again with a smaller step
                self.state = self.state.fixed_step(t_max - self.state.trv[0], **step_kwargs)
            else:
                self.state = state

            if 'inertial' in self.logs: self.log_inertial()
            if 'elements' in self.logs: self.log_elements()

        return


    def integrate_until(self, dt_requested, trigger, accuracy = 1e-6, gradient = False, **step_kwargs):
        dt     = dt_requested
        dt_min = dt
        dt_max = dt

        print "Iterating adaptively..."
        while trigger.waiting(self.state):
            state, dt = self.state.adaptive_step(dt, accuracy = accuracy, **step_kwargs)
            dt_min = min(dt, dt_min)
            dt_max = max(dt, dt_min)
            

            if trigger.overshot(state):
                break
            else:
                self.state = state

            print "time = ", self.state.trv[0], "dt = ", str(dt)

            if 'inertial' in self.logs: self.log_inertial()
            if 'elements' in self.logs: self.log_elements()
                
        print "Iterating with fixed steps..."
        self.fixed_integrate_until(dt_requested, trigger, gradient = gradient, **step_kwargs)

        # At this point, we should be within dt_requested of trigger, and not past it.

        return

    def fixed_integrate_until(self, dt, trigger, gradient = False, **step_kwargs):
        """Propagate by fixed step dt."""
        ii = 0
        while trigger.waiting(self.state):
            state = self.state.fixed_step(dt, **step_kwargs)
            
            if trigger.overshot(state): # don't record the state if we overshot
                print "overshot"
                break
            else:
                self.state = state

            print "time = ", state.trv[0], "dt = ", str(dt)
            
            if 'inertial' in self.logs: self.log_inertial()
            if 'elements' in self.logs: self.log_elements()

            ii += 1
        return

    def log_inertial(self):
        return self.logs['inertial'].write(self.state.trv)

    def log_elements(self):
        return self.logs['elements'].write(np.hstack((self.state.trv[0], self.state.conic_elements()[0:6])))


