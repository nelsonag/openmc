#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False

import numpy as np

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


    cdef double eval_cm(self, double mu_l, double Eo_l, double c, double Ein):
        cdef double J, Eo_cm, mu_cm
        J = 1. / sqrt(1. + c * c - 2. * c * mu_l)
        Eo_cm = Eo_l / (J * J)
        mu_cm = J * (mu_l - c)
        return self.eval(mu_cm, Eo_cm)

    cpdef integrate_cm_legendre(self, double Ein, double[::1] Eouts, double awr,
                                int order):
        cdef int g, eo, l, orders, p
        cdef double Eout_cm_max, inv_awrp1, Eout_l_max, Eout_l_min, a, b
        cdef double c, mu_l_min, dE, dmu, u, yg, f, Eout, Eout_prev, value, mu_lo
        cdef double[::1] y
        cdef double[::1] y_prev
        cdef np.ndarray[np.double_t, ndim=2] integral

        integral = np.zeros((Eouts.shape[0] - 1, order + 1))

        Eout_cm_max = self.Eout_max()

        inv_awrp1 = 1. / (awr + 1.)
        Eout_l_min = max(Ein * inv_awrp1 * inv_awrp1 / _MIN_C2, Eouts[0])
        Eout_l_max = min(Ein * (sqrt(Eout_cm_max / Ein) + inv_awrp1)**2,
                         Eouts[Eouts.shape[0] - 1])

        dE = (Eout_l_max - Eout_l_min) / (_N_EOUT_DOUBLE - 1.)

        # Find the group of Eout_l_min
        g = np.searchsorted(Eouts, Eout_l_min) - 1

        Eout = Eout_l_min - dE
        y = np.zeros(integral.shape[1])
        y_prev = np.zeros_like(y)
        for eo in range(_N_EOUT):
            Eout_prev = Eout
            if eo == _N_EOUT - 1:
                Eout = Eouts[integral.shape[0]]
            else:
                Eout += dE
            y_prev = y
            c = inv_awrp1 * sqrt(Ein / Eout)
            mu_l_min = (1. + c * c - Eout_cm_max / Eout) / (2. * c)

            # Make sure we stay in the allowed bounds
            if mu_l_min < -1.:
                mu_l_min = -1.
            elif mu_l_min > 1.:
                break

            dmu = 1. - mu_l_min
            for p in range(_N_QUAD):
                u = _POINTS[p] * dmu + mu_l_min
                _FMU[p] = _WEIGHTS[p] * self.eval_cm(u, Eout, c, Ein)

            for l in range(order + 1):
                y[l] = 0.
                for p in range(_N_QUAD):
                    u = _POINTS[p] * dmu + mu_l_min
                    y[l] += _FMU[p] * eval_legendre(l, u)
                y[l] *= dmu

            if eo > 0:
                # Perform the running integration of our point according to the
                # outgoing group of Eout

                # First, if the Eout point is above the next group boundary, then
                # we are straddling the line and we only include the portion for the
                # current group
                if Eout > Eouts[g + 1]:
                    # Include the integral up to the group boundary only
                    f = (Eouts[g + 1] - Eout_prev) / (Eout - Eout_prev)
                    a = 0.5 * (Eouts[g + 1] - Eout_prev)
                    b = 0.5 * (Eout - Eouts[g + 1])
                    for l in range(integral.shape[1]):
                        yg = (1. - f) * y[l] + f * y_prev[l]
                        integral[g, l] += a * (y_prev[l] + yg)
                        # And add the top portion to the next group
                        integral[g + 1, l] += b * (yg + y[l])
                    # Move our group counter to the next group
                    g += 1
                else:
                    # Then there is no group straddling, so just do a standard
                    # trapezoidal integration
                    a = 0.5 * (Eout - Eout_prev)
                    for l in range(integral.shape[1]):
                        integral[g, l] += a * (y_prev[l] + y[l])

        return integral


    cpdef integrate_cm_histogram(self, double Ein, double[::1] Eouts,
                                 double awr, double[::1] mus):
        cdef int g, eo, l, orders, p
        cdef double Eout_cm_max, inv_awrp1, Eout_l_max, Eout_l_min, a, b
        cdef double c, mu_l_min, dE, dmu, u, yg, f, Eout, Eout_prev, value, mu_lo
        cdef double[::1] y
        cdef double[::1] y_prev
        cdef np.ndarray[np.double_t, ndim=2] integral

        integral = np.zeros((Eouts.shape[0] - 1, mus.shape[0] - 1))
        Eout_cm_max = self.Eout_max()
        orders = integral.shape[1]

        inv_awrp1 = 1. / (awr + 1.)
        Eout_l_min = max(Ein * inv_awrp1 * inv_awrp1 / _MIN_C2, Eouts[0])
        Eout_l_max = min(Ein * (sqrt(Eout_cm_max / Ein) + inv_awrp1)**2,
                         Eouts[Eouts.shape[0] - 1])

        dE = (Eout_l_max - Eout_l_min) / (_N_EOUT_DOUBLE - 1.)

        # Find the group of Eout_l_min
        g = np.searchsorted(Eouts, Eout_l_min) - 1

        Eout = Eout_l_min - dE
        y = np.zeros(integral.shape[1])
        y_prev = np.zeros_like(y)
        for eo in range(_N_EOUT):
            Eout_prev = Eout
            if eo == _N_EOUT - 1:
                Eout = Eouts[integral.shape[0]]
            else:
                Eout += dE
            y_prev = y
            c = inv_awrp1 * sqrt(Ein / Eout)
            mu_l_min = (1. + c * c - Eout_cm_max / Eout) / (2. * c)

            # Make sure we stay in the allowed bounds
            if mu_l_min < -1.:
                mu_l_min = -1.
            elif mu_l_min > 1.:
                break

            for l in range(mus.shape[0] - 1):
                value = 0.
                mu_lo = max(mus[l], mu_l_min)

                dmu = mus[l + 1] - mu_lo
                for p in range(_N_QUAD):
                    u = _POINTS[p] * dmu + mu_lo
                    value += _WEIGHTS[p] * self.eval_cm(u, Eout, c, Ein)
                y[l] = value * dmu

            if eo > 0:
                # Perform the running integration of our point according to the
                # outgoing group of Eout

                # First, if the Eout point is above the next group boundary, then
                # we are straddling the line and we only include the portion for the
                # current group
                if Eout > Eouts[g + 1]:
                    # Include the integral up to the group boundary only
                    f = (Eouts[g + 1] - Eout_prev) / (Eout - Eout_prev)
                    a = 0.5 * (Eouts[g + 1] - Eout_prev)
                    b = 0.5 * (Eout - Eouts[g + 1])
                    for l in range(integral.shape[1]):
                        yg = (1. - f) * y[l] + f * y_prev[l]
                        integral[g, l] += a * (y_prev[l] + yg)
                        # And add the top portion to the next group
                        integral[g + 1, l] += b * (yg + y[l])
                    # Move our group counter to the next group
                    g += 1
                else:
                    # Then there is no group straddling, so just do a standard
                    # trapezoidal integration
                    a = 0.5 * (Eout - Eout_prev)
                    for l in range(integral.shape[1]):
                        integral[g, l] += a * (y_prev[l] + y[l])

        return integral

    cpdef integrate_lab_legendre(self, double[::1] Eouts, int order,
                                 double[::1] mus_grid, double[:, ::1] wgts):
        pass

    cpdef integrate_lab_histogram(self, double[::1] Eouts, double[::1] mus):
        pass
