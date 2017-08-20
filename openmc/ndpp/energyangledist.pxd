#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False


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
