#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False


from openmc.stats.bisect cimport bisect, bisect_int
from .cevals cimport *
from .energyangledist cimport EnergyAngle_Cython


cdef class Correlated(EnergyAngle_Cython):
    """ Class to contain the data and methods for a correlated distribution;
    """
    cdef list adists

    cdef double eval(self, double mu, double Eout)
