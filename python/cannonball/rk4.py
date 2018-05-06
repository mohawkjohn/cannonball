"""
Source: http://www.physics.buffalo.edu/phy410-505/2011/topic2/app1/index.html
"""

import numpy as np

def step(x_p, dh, flow, gradient = False, **flow_args):
    '''replaces x(h) by x(h + dh)'''
    if gradient:
        k1, da_dr = flow(x_p, gradient = True, **flow_args) * dh
    else:
        k1 = flow(x_p, **flow_args) * dh
        da_dr = None
    x_temp = x_p + k1 / 2
    k2 = flow(x_temp, **flow_args) * dh
    x_temp = x_p + k2 / 2
    k3 = flow(x_temp, **flow_args) * dh
    x_temp = x_p + k3
    k4 = flow(x_temp, **flow_args) * dh

    x_n = x_p + (k1 + k2*2 + k3*2 + k4) / 6.0
    return (x_n, da_dr)
        

def adaptive_step(x_p, dt, flow, max_dt = None, accuracy=1e-6, gradient = False, **flow_args):  # from Numerical Recipes
    SAFETY = 0.9; PGROW = -0.2; PSHRINK = -0.25;
    ERRCON = 1.89E-4; TINY = 1.0E-30
    scale = flow(x_p)
    scale = np.absolute(x_p) + np.absolute(scale * dt) + TINY
    
    x_half = None
    
    while True:
        dt /= 2
        # Take two half steps
        if gradient:
            x_half, da_dr = step(x_p, dt, flow, gradient = True, **flow_args)
        else:
            x_half, temp = step(x_p, dt, flow, **flow_args)
        x_half, temp = step(x_half, dt, flow, **flow_args)
        dt *= 2

        # Take one full step
        x_full, temp = step(x_p, dt, flow, **flow_args)

        dx = x_half - x_full
        error = np.absolute(dx / scale).max() / accuracy
        
        if error <= 1:
            break;
        dt_temp = SAFETY * dt * error**PSHRINK
        if dt >= 0:
            dt = max(dt_temp, 0.1 * dt)
        else:
            dt = min(dt_temp, 0.1 * dt)
        if abs(dt) == 0.0:
            raise OverflowError("step size underflow")
        
    if error > ERRCON:
        dt *= SAFETY * error**PGROW
    else:
        dt *= 5

    if gradient:
        return (dt, x_half + dx / 15.0, da_dr)
    else:
        return (dt, x_half + dx / 15.0, None)
