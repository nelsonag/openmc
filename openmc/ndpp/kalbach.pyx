#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: fast_gil=True

from libc.math cimport sqrt, sinh, cosh

from openmc.stats.bisect cimport bisect, bisect_int
from .cevals cimport *


cdef class KM:
    """ Class to contain the data and methods for a Kalbach-Mann distribution;
    """
    cdef double[:] slope_x
    cdef double[:] slope_y
    cdef long[:] slope_breakpoints
    cdef long[:] slope_interpolation
    cdef double[:] precompound_y
    cdef double[:] edist_x
    cdef double[:] edist_p
    cdef int edist_interpolation

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

    def __call__(self, double mu, double Eout):
        # Compute f(Eout) * g(mu, Eout) for the KM distribution

        cdef double a, r, f_Eout
        cdef int interp_type
        cdef size_t idx
        a = tabulated1d_eval_w_search_params(self.slope_x, self.slope_y,
                                             self.slope_breakpoints,
                                             self.slope_interpolation, Eout,
                                             idx, interp_type)

        if a == 0.:
            return 0.
        else:
            r = fast_tabulated1d_eval(Eout, self.slope_x[idx],
                                      self.slope_x[idx + 1],
                                      self.precompound_y[idx],
                                      self.precompound_y[idx + 1], interp_type)

            f_Eout = tabular_eval(self.edist_x, self.edist_p,
                                  self.edist_interpolation, Eout)
            return f_Eout * 0.5 * a / sinh(a) * (cosh(a * mu) +
                                                 r * sinh(a * mu))

    cpdef get_domain(self, double Ein=0.):
        return self.edist_x[0], self.edist_x[self.edist_x.shape[0] - 1]
