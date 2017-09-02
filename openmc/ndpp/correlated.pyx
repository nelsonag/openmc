#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False

from openmc.stats import Tabular, Uniform, Discrete
import numpy as np


cdef class Correlated(EnergyAngle_Cython):
    """ Class to contain the data and methods for a correlated distribution;
    """

    def __init__(self, list adists, edist):
        cdef size_t num_adists = len(adists)
        cdef size_t max_x, i, j

        # Find the maximum dimension for the angular distribution data
        max_x = 0
        for i in range(num_adists):
            if len(adists[i]._x) > max_x:
                max_x = len(adists[i]._x)

        # Now store the data
        self.adists_x = cvarray(shape=(num_adists, max_x),
                                itemsize=sizeof(double), format="d")
        self.adists_p = cvarray(shape=(num_adists, max_x),
                                itemsize=sizeof(double), format="d")
        self.adists_type = cvarray(shape=(num_adists,), itemsize=sizeof(int),
                                   format="i")
        self.adists_dim = cvarray(shape=(num_adists,), itemsize=sizeof(int),
                                   format="i")

        for i in range(num_adists):
            self.adists_dim[i] = len(adists[i]._x)
            for j in range(self.adists_dim[i]):
                self.adists_x[i, j] = adists[i]._x[j]
                self.adists_p[i, j] = adists[i]._p[j]
            if isinstance(adists[i], Tabular):
                self.adists_type[i] = _ADIST_TYPE_TABULAR
            elif isinstance(adists[i], Uniform):
                self.adists_type[i] = _ADIST_TYPE_UNIFORM
            elif isinstance(adists[i], Discrete):
                self.adists_type[i] = _ADIST_TYPE_DISCRETE
            else:
                raise ValueError("Invalid Angular Distribution Type")

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

    def __call__(self, double mu, double Eout):
        return self.eval(mu, Eout)

    cdef double eval(self, double mu, double Eout):
        # Compute f(Eout) * g(mu, Eout) for the correlated distribution
        cdef double f, g
        cdef size_t i
        cdef double interpolant
        cdef int max_x

        f = tabular_eval_w_search_params(self.edist_x, self.edist_p,
                                         self.edist_x.shape[0] - 1,
                                         self.edist_interpolation, Eout, &i)

        # Pick the nearest angular distribution (consistent with OpenMC)
        # Make sure the Eout points are not the same value (this happens in a
        # few evaluations). If they are the same, just use the lower
        # distribution.
        interpolant = (self.edist_x[i + 1] - self.edist_x[i])
        if interpolant > 0.:
            # Now that we made sure the consective Eout points are not the same
            # continue finding the interpolant to see which distribution we
            # are closer to
            interpolant = (Eout - self.edist_x[i]) / interpolant
            if interpolant > 0.5:
                i += 1

        # Now find the angular distribution value
        max_x = self.adists_dim[i] - 1
        if self.adists_type[i] == _ADIST_TYPE_TABULAR:
            g = tabular_eval(self.adists_x[i, :], self.adists_p[i, :],
                             max_x, LINLIN, mu)
        elif self.adists_type[i] == _ADIST_TYPE_UNIFORM:
            g = 0.5
        elif self.adists_type[i] == _ADIST_TYPE_DISCRETE:
            g = discrete_eval(self.adists_x[i, :], self.adists_p[i, :], max_x,
                              mu)

        return f * g

    cpdef integrate_lab_legendre(self, double[::1] Eouts,
                                 double[:, ::1] integral, double [::1] grid,
                                 double[::1] mus_grid, double[:, ::1] wgts):
        """Routine to integrate this distribution represented as a lab-frame
        and expanded via Legendre polynomials
        """

        cdef int g, eo, l, p
        cdef double Eout_hi, Eout_lo, Eo_min, Eo_max
        cdef double mu_l_min, dE, Eout, dmu, u, value

        Eo_min = self.Eout_min()
        Eo_max = self.Eout_max()

        for g in range(Eouts.shape[0] - 1):
            Eout_lo = Eouts[g]
            Eout_hi = Eouts[g + 1]

            # If our group is below the possible outgoing energies, just skip it
            if Eout_hi < Eo_min:
                continue
            # If our group is above the max energy then we are all done
            if Eout_lo > Eo_max:
                break

            Eout_lo = max(Eo_min, Eout_lo)
            Eout_hi = min(Eo_max, Eout_hi)

            dE = (Eout_hi - Eout_lo) / (_N_EOUT_DOUBLE - 1.)

            for eo in range(_N_EOUT):
                Eout = Eout_lo + ((<double>eo) * dE)

                for p in range(mus_grid.shape[0]):
                    value = self.eval(mus_grid[p], Eout)
                    for l in range(grid.shape[0]):
                        grid[l] = value * wgts[p, l]

                for l in range(integral.shape[1]):
                    integral[g, l] += _SIMPSON_WEIGHTS[eo] * grid[l]
            for l in range(integral.shape[1]):
                integral[g, l] *= dE

    cpdef integrate_lab_histogram(self, double[::1] Eouts,
                                  double[:, ::1] integral, double [::1] grid,
                                  double[::1] mus):
        """Routine to integrate this distribution represented as a lab-frame
        and expanded in the histogram bins defined by mus
        """

        cdef int g, eo, l, p
        cdef double Eout_hi, Eout_lo, Eo_min, Eo_max
        cdef double mu_l_min, dE, Eout, dmu, u, value

        Eo_min = self.Eout_min()
        Eo_max = self.Eout_max()

        for g in range(Eouts.shape[0] - 1):
            Eout_lo = Eouts[g]
            Eout_hi = Eouts[g + 1]

            # If our group is below the possible outgoing energies, just skip it
            if Eout_hi < Eo_min:
                continue
            # If our group is above the max energy then we are all done
            if Eout_lo > Eo_max:
                break

            Eout_lo = max(Eo_min, Eout_lo)
            Eout_hi = min(Eo_max, Eout_hi)

            dE = (Eout_hi - Eout_lo) / (_N_EOUT_DOUBLE - 1.)

            for eo in range(_N_EOUT):
                Eout = Eout_lo + ((<double>eo) * dE)

                for l in range(mus.shape[0] - 1):
                    value = 0.
                    dmu = mus[l + 1] - mus[l]
                    for p in range(_N_QUAD):
                        u = _POINTS[p] * dmu + mus[l]
                        value += _WEIGHTS[p] * self.eval(u, Eout)
                    grid[l] = value * dmu

                for l in range(integral.shape[1]):
                    integral[g, l] += _SIMPSON_WEIGHTS[eo] * grid[l]
            for l in range(integral.shape[1]):
                integral[g, l] *= dE
