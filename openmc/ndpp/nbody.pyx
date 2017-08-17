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
