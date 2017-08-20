#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False


from cython.view cimport array as cvarray

from openmc.stats.bisect cimport bisect, bisect_int
from .cevals cimport *
from .energyangledist cimport EnergyAngle_Cython

cdef int _ADIST_TYPE_TABULAR = 0
cdef int _ADIST_TYPE_UNIFORM = 1
cdef int _ADIST_TYPE_DISCRETE = 2


cdef class Correlated(EnergyAngle_Cython):
    """ Class to contain the data and methods for a correlated distribution;
    """
    cdef double[:, ::1] adists_x
    cdef double[:, ::1] adists_p
    cdef int[::1] adists_type
    cdef int[::1] adists_dim

    cdef double eval(self, double mu, double Eout)
