#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False


cdef class EnergyAngle_Cython:
    """ Class to contain the data and methods for a Kalbach-Mann,
    Correlated, Uncorrelated, and NBody distribution;
    """

    cpdef double Eout_min(self):
        return self.edist_x[0]

    cpdef double Eout_max(self):
        return self.edist_x[self.edist_x.shape[0] - 1]

    cdef double eval(self, double mu, double Eout):
        pass
