#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False

import numpy as np

cdef class Uncorrelated(EnergyAngle_Cython):
    """ Class to contain the data and methods for an uncorrelated distribution;
    """

    def __init__(self, adist, edist):
        self.adist = adist
        self.edist = edist
        self.edist_x = edist._x
        self.edist_p = edist._p
        if edist._interpolation == 'histogram':
            self.edist_interpolation = HISTOGRAM
        elif edist._interpolation == 'linear-linear':
            self.edist_interpolation = LINLIN
        elif edist._interpolation == 'linear-log':
            self.edist_interpolation = LINLOG
        elif edist._interpolation == 'log-linear':
            self.edist_interpolation = LOGLIN
        elif edist._interpolation == 'log-log':
            self.edist_interpolation = LOGLOG

        # Set the Eout min and max attributes
        self.Eout_min = self.edist_x[0]
        self.Eout_max = self.edist_x[self.edist_x.shape[0] - 1]

    def __call__(self, double mu, double Eout):
        return self.eval(mu, Eout)

    cdef double eval(self, double mu, double Eout):
        # Compute f(Eout) * g(mu) for the correlated distribution

        cdef double f_Eout

        f_Eout = tabular_eval(self.edist_x, self.edist_p,
                              self.edist_x.shape[0] - 1,
                              self.edist_interpolation, Eout)

        return f_Eout * self.adist._eval(mu)

    cpdef integrate_lab_legendre(self, double[::1] Eouts, int order,
                                 double[::1] mus_grid, double[:, ::1] wgts):
        """Routine to integrate this distribution represented as a lab-frame
        and expanded via Legendre polynomials
        """

        cdef int g, eo, l, p
        cdef double Eout_hi, Eout_lo
        cdef double mu_l_min, dE, Eout, dmu, u, value
        cdef double[::1] angle_integral
        cdef np.ndarray[np.double_t, ndim=2] integral

        integral = np.zeros((Eouts.shape[0] - 1, order + 1))

        angle_integral = np.empty(order + 1)

        for p in range(mus_grid.shape[0]):
            value = self.adist._eval(mus_grid[p])
            for l in range(angle_integral.shape[0]):
                angle_integral[l] = value * wgts[p, l]

        for g in range(Eouts.shape[0] - 1):
            Eout_lo = Eouts[g]
            Eout_hi = Eouts[g + 1]

            # If our group is below the possible outgoing energies, just skip it
            if Eout_hi < self.Eout_min:
                continue
            # If our group is above the max energy then we are all done
            if Eout_lo > self.Eout_max:
                break

            Eout_lo = max(self.Eout_min, Eout_lo)
            Eout_hi = min(self.Eout_max, Eout_hi)

            for l in range(angle_integral.shape[0]):
                integral[g, l] = angle_integral[l] * \
                    self.edist.integrate(Eout_lo, Eout_hi)

        return integral

    cpdef integrate_lab_histogram(self, double[::1] Eouts, double[::1] mus):
        """Routine to integrate this distribution represented as a lab-frame
        and expanded in the histogram bins defined by mus
        """

        cdef int g, eo, l, p
        cdef double Eout_hi, Eout_lo
        cdef double mu_l_min, dE, Eout, dmu, u, value
        cdef double[::1] angle_integral
        cdef np.ndarray[np.double_t, ndim=2] integral

        integral = np.zeros((Eouts.shape[0] - 1, mus.shape[0] - 1))

        angle_integral = np.empty(mus.shape[0] - 1)

        for l in range(mus.shape[0] - 1):
            value = 0.
            dmu = mus[l + 1] - mus[l]
            for p in range(_N_QUAD):
                u = _POINTS[p] * dmu + mus[l]
                value += _WEIGHTS[p] * self.adist._eval(u)
            angle_integral[l] = value * dmu

        for g in range(Eouts.shape[0] - 1):
            Eout_lo = Eouts[g]
            Eout_hi = Eouts[g + 1]

            # If our group is below the possible outgoing energies, just skip it
            if Eout_hi < self.Eout_min:
                continue
            # If our group is above the max energy then we are all done
            if Eout_lo > self.Eout_max:
                break

            Eout_lo = max(self.Eout_min, Eout_lo)
            Eout_hi = min(self.Eout_max, Eout_hi)

            for l in range(angle_integral.shape[0]):
                integral[g, l] = angle_integral[l] * \
                    self.edist.integrate(Eout_lo, Eout_hi)

        return integral

