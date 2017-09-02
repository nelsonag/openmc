#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False

from libc.math cimport sqrt

cimport numpy as np
from scipy.special.cython_special cimport eval_legendre

from .cconstants cimport *

cdef class EnergyAngle_Cython:
    """ Class to contain the data and methods for a Kalbach-Mann,
    Correlated, Uncorrelated, and NBody distribution;
    """

    cdef double[::1] edist_x
    cdef double[::1] edist_p
    cdef int edist_interpolation

    cpdef double Eout_min(self)

    cpdef double Eout_max(self)

    cdef double eval(self, double mu, double Eout)

    cdef double eval_cm(self, double mu_l, double Eo_l, double c, double Ein)

    cpdef integrate_cm_legendre(self, double Ein, double[::1] Eouts, double awr,
                                int order)

    cpdef integrate_cm_histogram(self, double Ein, double[::1] Eouts,
                                 double awr, double[::1] mus)

    cpdef integrate_lab_legendre(self, double[::1] Eouts, int order,
                                 double[::1] mus_grid, double[:, ::1] wgts)

    cpdef integrate_lab_histogram(self, double[::1] Eouts,  double[::1] mus)
