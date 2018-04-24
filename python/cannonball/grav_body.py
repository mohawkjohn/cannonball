class GravBody(object):
    def semimajor_axis(self):
        return self.A

    def mu(self):
        return self.MU

class Earth(object):
    A    = 6378137.0 # m
    MU   = 3.986004415e14 # m^3 / s^2
    J2   = 0.0010826269
