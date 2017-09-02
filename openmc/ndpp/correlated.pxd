#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False


from cython.view cimport array as cvarray

from openmc.stats.bisect cimport bisect, bisect_int
from .cconstants cimport *
from .cevals cimport *
from .energyangledist cimport EnergyAngle_Cython

cdef class Correlated(EnergyAngle_Cython):
    """ Class to contain the data and methods for a correlated distribution;
    """
    cdef double[:, ::1] adists_x
    cdef double[:, ::1] adists_p
    cdef int[::1] adists_type
    cdef int[::1] adists_dim

    cdef double eval(self, double mu, double Eout)

    cpdef integrate_lab_legendre(self, double[::1] Eouts, int order,
                                 double[::1] mus_grid, double[:, ::1] wgts)

    cpdef integrate_lab_histogram(self, double[::1] Eouts, double[::1] mus)
