#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False

from libc.math cimport sqrt

cimport numpy as np
from scipy.special.cython_special cimport eval_legendre

from .cconstants cimport *
from .freegas cimport CALC_ER_INTEGRAL, calc_Er_integral_doppler, \
                      calc_Er_integral_cxs


cdef class TwoBody_FGK:
    """ Class to contain the data and methods for a Target-at-Rest Two-Body
    distribution
    """

    cdef public int adist_interp
    cdef public double[::1] adist_x
    cdef public double[::1] adist_p
    cdef public double[::1] xs_x
    cdef public double[::1] xs_y
    cdef public long[::1] xs_bpts
    cdef public long[::1] xs_interp
    cdef public double beta
    cdef public double half_beta_2
    cdef public double alpha
    cdef public double awr
    cdef public double kT
    cdef public double Eout_min
    cdef public double Eout_max
    cdef CALC_ER_INTEGRAL freegas_function

    cpdef integrate_cm_legendre(self, double Ein, double[::1] Eouts, int order,
                                double[::1] mus_grid, double[:, ::1] wgts)

    cpdef integrate_cm_histogram(self, double Ein, double[::1] Eouts,
                                 double[::1] mus)
