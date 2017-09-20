#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False


cdef class KM(EnergyAngle_Cython):
    """ Class to contain the data and methods for a Kalbach-Mann distribution;
    """

    def __init__(self, slope, precompound, edist):
        # Set our slope, precompound, and edist data

        self.slope_x = slope._x
        self.slope_y = slope._y
        self.slope_breakpoints = slope._breakpoints
        self.slope_interpolation = slope._interpolation

        # The KM slope and precompound have the same x grid, breakpoints, and
        # interpolation rules, so we dont need to store again
        self.precompound_y = precompound._y

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

        # Set the Eout min and max attributes
        self.Eout_min = self.edist_x[0]
        self.Eout_max = self.edist_x[self.edist_x.shape[0] - 1]

    def __call__(self, double mu, double Eout):
        return self.eval(mu, Eout)

    cdef double eval(self, double mu, double Eout):
        # Compute f(Eout) * g(mu, Eout) for the KM distribution

        cdef double a, r, f_Eout
        cdef int interp_type
        cdef size_t idx
        a = tabulated1d_eval_w_search_params(self.slope_x, self.slope_y,
                                             self.slope_x.shape[0] - 1,
                                             self.slope_breakpoints,
                                             self.slope_interpolation, Eout,
                                             &idx, &interp_type)

        if a == 0.:
            return 0.
        else:
            r = fast_tabulated1d_eval(Eout, self.slope_x[idx],
                                      self.slope_x[idx + 1],
                                      self.precompound_y[idx],
                                      self.precompound_y[idx + 1], interp_type)

            f_Eout = tabular_eval(self.edist_x, self.edist_p,
                                  self.edist_x.shape[0] - 1,
                                  self.edist_interpolation, Eout)
            return f_Eout * 0.5 * a / sinh(a) * (cosh(a * mu) +
                                                 r * sinh(a * mu))
