from .context import *

from cannonball import Orbiter
from cannonball.log import Log

class TestOrbiter(unittest.TestCase):

    def setUp(self):
        self.trv = np.array([0.0,
                            7000000.0, 0.0,    0.0,
                            0.0,       4000.0, 0.0])
        
    def test_integrate_log(self):
        """Integrates, stores, and reads a trajectory"""
        trv  = self.trv
        max_time = 20.0
        dt_guess = 0.5

        orb  = Orbiter(trv)
        orb.integrate(dt_guess, max_time)
        orb.log.close()
        
        log  = Log('trv.log')
        data = log.read()
        np.testing.assert_equal(data[0,0], 0.0)
        np.testing.assert_equal(data[0,1], dt_guess)
        np.testing.assert_equal(data[0,-1], max_time)


if __name__ == '__main__':
    unittest.main()
