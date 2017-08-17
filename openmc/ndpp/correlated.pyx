#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False


cdef class Correlated(EnergyAngle_Cython):
    """ Class to contain the data and methods for a correlated distribution;
    """

    def __init__(self, list adists, edist):
        self.adists = adists
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
        # Compute f(Eout) * g(mu, Eout) for the correlated distribution

        cdef double f_Eout
        cdef size_t i
        cdef double interpolant

        f_Eout = tabular_eval_w_search_params(self.edist_x, self.edist_p,
                                              self.edist_interpolation, Eout,
                                              &i)

        # Pick the nearest angular distribution (consistent with OpenMC)
        # Make sure the Eout points are not the same value (this happens in a
        # few evaluations). If they are the same, just use the lower
        # distribution.
        interpolant = (self.edist_x[i + 1] - self.edist_x[i])
        if interpolant > 0.:
            # Now that we made sure the consective Eout points are not the same
            # continue finding the interpolant to see which distribution we
            # are closer to
            interpolant = (Eout - self.edist_x[i]) / interpolant
            if interpolant > 0.5:
                i += 1

        # Now evaluate and return our resultant angular distribution
        return f_Eout * self.adists[i]._eval(mu)
