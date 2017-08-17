#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False


cdef class Uncorrelated(EnergyAngle_Cython):
    """ Class to contain the data and methods for an uncorrelated distribution;
    """

    def __init__(self, adist, edist):
        self.adist = adist
        self.edist = edist
        self.edist_x = edist._x
        self.edist_p = edist._p
        if edist._interpolation == 'histogram':
            self.edist_interpolation = HISTOGRAM
        elif edist._interpolation == 'linear-linear':
            self.edist_interpolation = LINLIN
        elif edist._interpolation == 'linear-log':
            self.edist_interpolation = LINLOG
        elif edist._interpolation == 'log-linear':
            self.edist_interpolation = LOGLIN
        elif edist._interpolation == 'log-log':
            self.edist_interpolation = LOGLOG

    def __call__(self, double mu, double Eout):
        return self.eval(mu, Eout)

    cdef double eval(self, double mu, double Eout):
        # Compute f(Eout) * g(mu) for the correlated distribution

        cdef double f_Eout

        f_Eout = tabular_eval(self.edist_x, self.edist_p,
                              self.edist_interpolation, Eout)

        return f_Eout * self.adist._eval(mu)
