#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: fast_gil=True

from libc.math cimport sqrt

cimport numpy as np
import numpy as np

_PI = np.pi


cdef class NBody:
    """ Class to contain the data and methods for a Nbody reaction; this is
    the only distribution type with a class because it does not have a
    pre-initialized edist and adist function available.
    """
    cdef public double Emax, Estar, C, exponent

    cpdef get_domain(self, double Ein=*)

    cdef double mu_min(self, double Eout)

    cdef double integrand_legendre_lab(self, double mu, double Eout)

    cdef integrand_histogram_lab(self, double Eout, double[::1] mus,
                                 double[::1] integral)
