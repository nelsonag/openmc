#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False

cimport numpy as np

from .cconstants cimport *
from .cevals cimport *
from .energyangledist cimport EnergyAngle_Cython


cdef class Uncorrelated(EnergyAngle_Cython):
    """ Class to contain the data and methods for an uncorrelated distribution;
    """
    cdef object adist
    cdef object edist

    cdef double eval(self, double mu, double Eout)

    cpdef integrate_lab_legendre(self, double[::1] Eouts, int order,
                                 double[::1] mus_grid, double[:, ::1] wgts)

    cpdef integrate_lab_histogram(self, double[::1] Eouts, double[::1] mus)
