#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False

from libc.math cimport sqrt, sinh, cosh

from openmc.stats.bisect cimport bisect, bisect_int
from .cevals cimport *
from .energyangledist cimport EnergyAngle_Cython


cdef class KM(EnergyAngle_Cython):
    """ Class to contain the data and methods for a Kalbach-Mann distribution;
    """
    cdef double[:] slope_x
    cdef double[:] slope_y
    cdef long[:] slope_breakpoints
    cdef long[:] slope_interpolation
    cdef double[:] precompound_y

    cdef double eval(self, double mu, double Eout)