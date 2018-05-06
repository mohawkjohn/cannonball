import rk4
from trigger import Trigger, TimeTrigger

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
                     conic_mu   = 3.986004415e14,
                     r_eq       = 6378137.0):
            self.conic_mu    = conic_mu
            self.r_eq        = r_eq

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

        def radial_velocity(self):
            """Compute the component of velocity inline with the radius vector
            from the body we're orbiting."""
            r = self.trv[1:4]
            v = self.trv[4:7]
            r_mag = norm(r)
            r_norm = r / r_mag
            return v.dot(r_norm)

        def radial_acceleration(self):
            r = self.trv[1:4]
            v = self.trv[4:7]
            a = gravity.basic_gravity(self.trv, r_eq = self.r_eq, mu = self.conic_mu)[4:7]
            r_mag = norm(r)
            return (v.dot(v) + r.dot(a)) / r_mag - (v.dot(r))**2 / (r_mag**3)

        def jerk(self):
            r = self.trv[1:4]
            v = self.trv[4:7]
            r_mag = norm(r)
            j = self.conic_mu * (-v / r_mag + r * v.dot(r) / r_mag**3)
            return j

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

        def period(self):
            a = self.semimajor_axis()
            return 2.0 * np.pi * np.sqrt(a**3 / self.conic_mu)
        
        def adaptive_step(self, dt, **step_kwargs):
            dt, trv, temp = rk4.adaptive_step(self.trv, dt, flow = gravity.basic_gravity, mu = self.conic_mu, r_eq = self.r_eq, **step_kwargs)
            return Orbiter.State(trv = trv, conic_mu = self.conic_mu), dt

        def fixed_step(self, dt, **step_kwargs):
            trv, temp = rk4.step(self.trv, dt, flow = gravity.basic_gravity, mu = self.conic_mu, r_eq = self.r_eq, **step_kwargs)
            return Orbiter.State(trv = trv, conic_mu = self.conic_mu)
        
        def estimate_time_to_apsis(self, steps = 20, tolerance = 1e-5, max_dt = 200.0):
            """Extremely rough estimate, strictly for propagation purposes.
            Uses Euler's method to try to find the time to an apsis.
            However, because it's Euler's method, it can't really distinguish
            between the previous apsis and the next apsis. So, if you want
            to find the next apsis, you have to wait until the inflection point
            on the radial acceleration (the local extreme for the radial velocity
            just prior to the apsis of interest)."""

            x = self

            t = 0.0

            for ii in range(0, steps):
                vr = x.radial_velocity()
                ar = x.radial_acceleration()
                if np.abs(ar) < tolerance: # prevent divide by zero
                    return t
                dt  = -vr / ar

                # Have we solved it? Or is our t too large?
                if np.abs(dt) < tolerance:
                    return t
                elif dt > max_dt:    dt =  max_dt
                elif dt < -max_dt:   dt = -max_dt
                
                x = x.fixed_step(dt)
                t += dt

            return t

        def estimate_time_to_next_apsis(self, steps = 20, tolerance = 1e-5, max_dt = 200.0):
            """Estimate the time remaining until *next* apsis.
            
            Uses a combination of large time-step propagation and
            Euler's method to ensure the resulting time is positive and
            decreases more or less linearly.
            """
            total_t = 0.0
            x = self
            t = x.estimate_time_to_apsis(steps = 1, max_dt = max_dt, tolerance = tolerance)
            # If we didn't find a positive time estimate, move us forward by
            # max_dt, then estimate again.
            while t < 0.0:
                x = x.fixed_step(max_dt)
                total_t += max_dt
                t = x.estimate_time_to_apsis(steps = 1, max_dt = max_dt, tolerance = tolerance)

            # Once we've got a positive time, get a more precise estimate.
            t = x.estimate_time_to_apsis(steps = steps, max_dt = max_dt, tolerance = tolerance)
            return total_t + t
    
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
        trigger = TimeTrigger(t_max)
        return self.integrate_until(dt_requested, trigger, accuracy = accuracy, gradient = gradient, **step_kwargs)


    def integrate_until(self, dt_requested, trigger, accuracy = 1e-6, gradient = False, **step_kwargs):
        dt       = dt_requested
        dt_guess = dt_requested
        #dt_min = dt
        #dt_max = dt

        # Always allow at least one step, because otherwise we can get stuck
        # at singularities --- e.g., when we start at apoapsis and want to
        # integrate until a periapsis trigger, we find that the < condition
        # fails at the initial step.
        self.state, dt = self.state.adaptive_step(dt, accuracy = accuracy, **step_kwargs)
        if 'inertial' in self.logs: self.log_inertial()
        if 'elements' in self.logs: self.log_elements()        

        while self.state < trigger:

            state, dt = self.state.adaptive_step(dt, accuracy = accuracy, **step_kwargs)

            if trigger < state: # overshot
                break
            else:
                self.state = state

            if 'inertial' in self.logs: self.log_inertial()
            if 'elements' in self.logs: self.log_elements()
                
        self.fixed_integrate_until(dt_requested, trigger, gradient = gradient, at_least_once = False, **step_kwargs)

        # At this point, we should be within dt_requested of trigger, and not past it.
        if trigger.is_exact() and self.state < trigger:
            self.fixed_integrate_until(trigger.guess_remaining(self.state), trigger, gradient = gradient, at_least_once = False, **step_kwargs)

        return

    def fixed_integrate_until(self, dt, trigger, gradient = False, at_least_once = True, **step_kwargs):
        """Propagate by fixed step dt."""

        if at_least_once:
            self.state = self.state.fixed_step(dt, **step_kwargs)
            if 'inertial' in self.logs: self.log_inertial()
            if 'elements' in self.logs: self.log_elements()
                
        while self.state < trigger:
            state = self.state.fixed_step(dt, **step_kwargs)
            
            if trigger < state: # don't record the state if we overshot
                break
            else:
                self.state = state
            
            if 'inertial' in self.logs: self.log_inertial()
            if 'elements' in self.logs: self.log_elements()

        return

    def log_inertial(self):
        return self.logs['inertial'].write(self.state.trv)

    def log_elements(self):
        return self.logs['elements'].write(np.hstack((self.state.trv[0], self.state.conic_elements()[0:6])))


    def adjust_apoapsis(self, dh, guess_dv = 1.0):
        """We want to produce some apoapsis change dh. Try a DV and see where
        that takes us."""
        v = self.trv[4:7]
        v_mag = norm(v)
        uv = v / v_mag # unit direction of velocity change
        v1 = uv * (v_mag + guess_dv)
        test = self.copy()
        test.trv[4:7] = v1
        
