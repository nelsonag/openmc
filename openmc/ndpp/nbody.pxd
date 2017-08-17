#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False

from libc.math cimport sqrt

cimport numpy as np
import numpy as np

from .energyangledist cimport EnergyAngle_Cython

_PI = np.pi


cdef class NBody(EnergyAngle_Cython):
    """ Class to contain the data and methods for a Nbody reaction; this is
    the only distribution type with a class because it does not have a
    pre-initialized edist and adist function available.
    """
    cdef public double Emax, Estar, C, exponent

    cdef double eval(self, double mu, double Eout)

    cdef double mu_min(self, double Eout)

    cdef double integrand_legendre_lab(self, double mu, double Eout)

    cdef integrand_histogram_lab(self, double Eout, double[::1] mus,
                                 double[::1] integral)

    cpdef double Eout_min(self)

    cpdef double Eout_max(self)
