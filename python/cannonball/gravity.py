import math

import numpy as np

def basic_gravity(trv,
                  step_using_index = False,
                  j2               = None,
                  r_eq             = 6378137.0,
                  mu               = 3.986004415e14,
                  gradient         = False):
    r     = trv[1:4]
    r_mag = math.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
    a     = r * -mu / r_mag**3

    if j2 is not None:
        r5 = r_mag**5
        r2 = r_mag**2
        k  = 3.0 * mu * j2 * r_eq**2 / (2.0 * r5)
        z2 = trv[3]**2
        a[0] += k * (5*z2/r2 - 1.0) * trv[1]
        a[1] += k * (5*z2/r2 - 1.0) * trv[2]
        a[2] += k * (5*z2/r2 - 3.0) * trv[3]

    if gradient: # compute change in acceleration with respect to change in position
        r = r.reshape((3,1))
        da_dr = r.dot(r.T) * 3.0 * mu / r5 - np.identity(3) * mu / r3
        
    flow = np.hstack(([1], trv[4:7], a))
    if step_using_index:
        flow /= trv[step_using_index] # change IV from t to vy

    if gradient:
        return flow, da_dr
    else:
        return flow
