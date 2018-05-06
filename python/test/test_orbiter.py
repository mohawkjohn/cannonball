from .context import *

from cannonball import Orbiter, TimeTrigger, ApsisTrigger
from cannonball.log import Log

from scipy.linalg import norm

PLOT = True
if PLOT:
    import matplotlib.pyplot as plt

class TestOrbiter(unittest.TestCase):

    def setUp(self):
        self.trv = np.array([0.0,
                            7000000.0, 0.0,    0.0,
                            0.0,       6500.0, 0.0])


    def test_radial_velocity_and_acceleration(self):
        """Radial velocity and acceleration are computed correctly"""
        trv = self.trv
        trigger = TimeTrigger
        dt_guess = 0.5

        t = []
        r  = []
        rv = []
        ra = []
        dt = []
        
        orb = Orbiter(trv = trv)
        while orb.state.trv[0] < 5000:
            
            orb.state = orb.state.fixed_step(10.0)

            v = orb.state.radial_velocity()
            a = orb.state.radial_acceleration()
            dt_est = orb.state.estimate_time_to_next_apsis()
            dt.append(dt_est)
            t.append(orb.state.trv[0])
            r.append(norm(orb.state.trv[1:4]))
            rv.append(v)
            ra.append(a)

        if 0 and PLOT:
            fig = plt.figure()
            ax0 = fig.add_subplot(411)
            ax0.plot(t, r)
            ax0.set_title("Radius (m)")
            ax0.grid(True)

            ax = fig.add_subplot(412, sharex=ax0)
            ax.plot(t, rv)
            ax.set_title("Radial velocity (m/s)")
            ax.grid(True)
            
            ax = fig.add_subplot(413, sharex=ax0)
            ax.plot(t, ra)
            ax.set_title("Radial acceleration (m/s/s)")
            ax.grid(True)

            ax = fig.add_subplot(414, sharex=ax0)
            ax.plot(t, dt)
            ax.set_title("Estimated time to next apsis")
            ax.grid(True)
            plt.show()
        
    def test_conic_elements(self):
        """Computes conic elements using SPICE """
        trv = self.trv
        orb = Orbiter(trv = trv)
        elements = orb.state.conic_elements()

    def test_time_trigger(self):
        """Integrates until some amount of time has passed."""
 
        trv      = self.trv
        dt_guess = 0.5
        trigger  = TimeTrigger(50 * 60)

        orb = Orbiter(trv = trv)
        orb.integrate_until(dt_guess, trigger)

        #self.assertLessEqual(orb.state.trv[0], trigger.time)
        #self.assertLessEqual(trigger.time - orb.state.trv[0], dt_guess)
        self.assertEqual(orb.state.trv[0], trigger.time)

    def test_apsis_trigger(self):
        """Integrates until apsis is reached."""

        trv      = self.trv
        dt_guess = 0.5
        trigger  = ApsisTrigger('periapsis')

        orb = Orbiter(trv = trv)
        orb.integrate_until(dt_guess, trigger)
        
        #self.assertAlmostEqual(orb.state.radial_velocity(), 0.0)
        self.assertNotEqual(orb.state.trv[0], 0.0)
        self.assertAlmostEqual(norm(orb.state.trv[1:4]) / orb.state.r_periapsis(), 1.0, places=6)

        # Make sure it didn't go past apsis.
        t_remaining = trigger.guess_remaining(orb.state)
        self.assertGreater(dt_guess, t_remaining)
        self.assertLessEqual(0.0, t_remaining)
        
    def test_integrate_log_and_elements(self):
        """Integrates, stores, and reads a trajectory"""
        # Also, plots trajectory and orbital elements for visual
        # verification.
        
        trv  = self.trv
        max_time = 500 * 60.0
        dt_guess = 0.5

        orb  = Orbiter(trv = trv)
        orb.integrate(dt_guess, max_time)
        for log in orb.logs:
            orb.logs[log].close()
        
        inrtl_log  = Log('inertial.log')
        inrtl = inrtl_log.read()
        np.testing.assert_equal(inrtl[0,0], 0.0)
        np.testing.assert_equal(inrtl[0,1], dt_guess)
        np.testing.assert_equal(inrtl[0,-1], max_time)

        elts_log = Log('elements.log')
        elts = elts_log.read()
        
        if 0 and PLOT:
            fig = plt.figure()
            ax = fig.add_subplot(321)
            ax.set_title("Inertial position over time")
            ax.plot(inrtl[0,:], inrtl[1,:])
            ax.plot(inrtl[0,:], inrtl[2,:])
            ax.plot(inrtl[0,:], inrtl[3,:])
            ax.set_ylabel("meters")
            ax.grid(True)
            plt.setp(ax.get_xticklabels(), visible=False)
            ax0 = ax

            ax = fig.add_subplot(322, sharex=ax0, sharey=ax0)
            ax.set_title("Periapsis and apoapsis over time")
            ax.plot(elts[0,:], elts[1,:])
            ax.plot(elts[0,:], elts[1,:] * (elts[2,:] + 1.0) / (1.0 - elts[2,:]))
            ax.set_ylabel("meters")
            ax.grid(True)
            plt.setp(ax.get_xticklabels(), visible=False)            
            
            ax = fig.add_subplot(323, sharex=ax0)
            ax.set_title("Inertial velocity over time")
            ax.plot(inrtl[0,:], inrtl[4,:])
            ax.plot(inrtl[0,:], inrtl[5,:])
            ax.plot(inrtl[0,:], inrtl[6,:])
            ax.set_ylabel("meters/second")
            ax.grid(True)
            plt.setp(ax.get_xticklabels(), visible=False)

            ax = fig.add_subplot(324, sharex=ax0)
            ax.set_title("Element angles over time")
            ax.plot(elts[0,:], elts[3,:] * (180.0 / math.pi), label='inc')
            ax.plot(elts[0,:], elts[4,:] * (180.0 / math.pi), label='lnode')
            ax.plot(elts[0,:], elts[5,:] * (180.0 / math.pi), label='argp')
            ax.plot(elts[0,:], elts[6,:] * (180.0 / math.pi), label='m0')
            #ax.set_xlabel("Time (s)")
            ax.set_ylabel("degrees")
            ax.grid(True)

            ax = fig.add_subplot(325, sharex=ax0)
            ax.set_title("Eccentricity over time")
            ax.plot(elts[0,:], elts[2,:])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("eccentricity")
            
            # The above looks like it's increasing over time. I'm not
            # worried about it.  I think it's increasing because of
            # the adaptive step size in the rk4 integrator creating a
            # phase shift with a really long period.  You can examine
            # this effect if you zoom in on the top points in this
            # next plot:

            ax = fig.add_subplot(326, sharex=ax0)
            ax.set_title("Radius")
            ax.plot(inrtl[0,:], norm(inrtl[1:4,:], axis=0))
            ax.set_ylabel("meters")
            ax.set_xlabel("Time (s)")

            plt.show()


if __name__ == '__main__':
    unittest.main()
