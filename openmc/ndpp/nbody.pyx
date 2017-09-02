#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False


cdef class NBody(EnergyAngle_Cython):
    """ Class to contain the data and methods for a Nbody reaction.
    """

    def __init__(self, this, Ein):
        # Find the Emax, C, and exponent parameters
        Ap = this.total_mass
        self.Emax = (Ap - 1.) / Ap * \
            (this.atomic_weight_ratio / (this.atomic_weight_ratio + 1.) * \
             Ein + this.q_value)
        # Estar is also ECM in some of the ENDF/NJOY references
        self.Estar = Ein * (1. / (this.atomic_weight_ratio + 1.))

        if this.n_particles == 3:
            self.C = 4. / (_PI * (self.Emax * self.Emax))
            self.exponent = 0.5
        elif this.n_particles == 4:
            self.C = 105. / (32. * sqrt(self.Emax**7))
            self.exponent = 2
        elif this.n_particles == 5:
            self.C = 256. / (14. * _PI * (self.Emax**5))
            self.exponent = 3.5

    cdef double eval(self, double mu, double Eout):
        # Compute f(mu, Eout) for the nbody distribution
        return self.C * sqrt(Eout) * (self.Emax - Eout) ** self.exponent

    cpdef double Eout_min(self):
        return 0.

    cpdef double Eout_max(self):
        return self.Emax

    cdef double mu_min(self, double Eout):
        cdef double value
        # The minimum mu allowed is that which makes the inner brackets
        # of the lab-frame PDF = 0
        value = (self.Emax - self.Estar - Eout) / (2. * sqrt(self.Estar * Eout))
        # Make sure we are within range
        if value < -1.:
            value = -1.
        elif value > 1.:
            value = 1.

        return value

    cdef double integrand_legendre_lab(self, double mu, double Eout):
        return self.C * sqrt(Eout) * \
            (self.Emax - (self.Estar + Eout - 2. * mu *
                          sqrt(self.Estar * Eout)))**self.exponent

    cdef integrand_histogram_lab(self, double Eout, double[::1] mus,
                                 double[::1] integral):
        cdef double a, b, c, exp_p1, mu_lo, mu_hi, mu_min
        cdef int i

        a = self.C * sqrt(self.Eout)
        b = self.Estar + Eout
        d = 2. * sqrt(self.Estar * Eout)
        exp_p1 = self.exponent + 1.
        mu_min = self.mu_min(Eout)
        for i in range(len(mus) - 1):
            mu_lo = mus[i]
            if mu_lo < mu_min:
                mu_lo = mu_min
            mu_hi = mus[i + 1]
            if mu_lo > mu_hi:
                integral[i] = 0.
                continue
            integral[i] = a / (d * self.exponent + d) * \
                ((self.Emax - b + d * mu_hi)**exp_p1 -
                 (self.Emax - b + d * mu_lo)**exp_p1)

    cpdef integrate_lab_legendre(self, double[::1] Eouts, int order,
                                 double[::1] mus_grid, double[:, ::1] wgts):
        cdef int g, eo, l
        cdef double Eout_hi, Eout_lo, Eo_max
        cdef double mu_l_min, dE, Eout, dmu, u
        cdef np.ndarray[np.double_t, ndim=2] integral
        cdef np.ndarray[np.double_t, ndim=1] grid

        integral = np.zeros((Eouts.shape[0] - 1, order + 1))
        grid = np.empty(integral.shape[1])

        # The astute reader will notice that we dont check against Eo_min in this
        # routine like we would for the correlated and uncorrelated versions.
        # The reason is because Eo_min is 0 for an NBody, so no reason to check.
        Eo_max = self.Eout_max()

        for g in range(Eouts.shape[0] - 1):
            Eout_lo = Eouts[g]
            Eout_hi = Eouts[g + 1]

            # If our group is above the max energy then we are all done
            if Eout_lo > Eo_max:
                break

            Eout_hi = min(Eo_max, Eout_hi)

            dE = (Eout_hi - Eout_lo) / (_N_EOUT_DOUBLE - 1.)

            for eo in range(_N_EOUT):
                Eout = Eout_lo + ((<double>eo) * dE)
                mu_l_min = self.mu_min(Eout)

                # Make sure we stay in the allowed bounds
                if mu_l_min < -1.:
                    mu_l_min = -1.
                elif mu_l_min > 1.:
                    break

                dmu = 1. - mu_l_min
                for p in range(_N_QUAD):
                    u = _POINTS[p] * dmu + mu_l_min
                    _FMU[p] = \
                        _WEIGHTS[p] * self.integrand_legendre_lab(u, Eout)

                for l in range(integral.shape[1]):
                    grid[l] = 0.
                    for p in range(_N_QUAD):
                        u = _POINTS[p] * dmu + mu_l_min
                        grid[l] += _FMU[p] * eval_legendre(l, u)
                    grid[l] *= dmu

                for l in range(integral.shape[1]):
                    integral[g, l] += _SIMPSON_WEIGHTS[eo] * grid[l]
            for l in range(integral.shape[1]):
                integral[g, l] *= dE

        return integral

    cpdef integrate_lab_histogram(self, double[::1] Eouts, double[::1] mus):
        cdef int g, eo, l
        cdef double Eout_hi, Eout_lo, Eo_max
        cdef double mu_l_min, dE, Eout
        cdef np.ndarray[np.double_t, ndim=2] integral
        cdef np.ndarray[np.double_t, ndim=1] grid

        integral = np.zeros((Eouts.shape[0] - 1, mus.shape[0] - 1))
        grid = np.empty(integral.shape[1])

        # The astute reader will notice that we dont check against Eo_min in this
        # routine like we would for integrate_corr_lab and integrate_uncorr_lab.
        # The reason is because Eo_min is 0 for an NBody, so no reason to check.
        Eo_max = self.Eout_max()

        for g in range(Eouts.shape[0] - 1):
            Eout_lo = Eouts[g]
            Eout_hi = Eouts[g + 1]

            # If our group is above the max energy then we are all done
            if Eout_lo > Eo_max:
                break

            Eout_hi = min(Eo_max, Eout_hi)

            dE = (Eout_hi - Eout_lo) / (_N_EOUT_DOUBLE - 1.)

            for eo in range(_N_EOUT):
                Eout = Eout_lo + ((<double>eo) * dE)
                mu_l_min = self.mu_min(Eout)

                # Make sure we stay in the allowed bounds
                if mu_l_min < -1.:
                    mu_l_min = -1.
                elif mu_l_min > 1.:
                    break

                self.integrand_histogram_lab(Eout, mus, grid)

                for l in range(integral.shape[1]):
                    integral[g, l] += _SIMPSON_WEIGHTS[eo] * grid[l]
            for l in range(integral.shape[1]):
                integral[g, l] *= dE

        return integral
