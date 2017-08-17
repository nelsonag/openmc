#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False

from .cevals cimport *
from .energyangledist cimport EnergyAngle_Cython


cdef class Uncorrelated(EnergyAngle_Cython):
    """ Class to contain the data and methods for an uncorrelated distribution;
    """
    cdef object adist
    cdef object edist

    cdef double eval(self, double mu, double Eout)
