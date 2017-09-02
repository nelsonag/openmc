#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False

from libc.math cimport sqrt

cimport numpy as np
from scipy.special.cython_special cimport eval_legendre

from .cconstants cimport *

ctypedef double (*TO_LAB_PTR)(double R, double w)

cdef class TwoBody_TAR:
    """ Class to contain the data and methods for a Target-at-Rest Two-Body
    distribution
    """

    cdef object adist
    cdef double awr
    cdef double R
    cdef double q_value
    cdef TO_LAB_PTR to_lab

    cdef double eval(self, double w, int l)

    cpdef integrate_cm_legendre(self, double Ein, double[::1] Eouts, int order)

    cpdef integrate_cm_histogram(self, double Ein, double[::1] Eouts,
                                 double[::1] mus)
