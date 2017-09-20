#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False

import numpy as np

cdef double to_lab(double R, double w):
    return (1. + R * w) / sqrt(1. + R * R + 2. * R * w)


cdef double to_lab_R_is_1(double R, double w):
    cdef double u
    # There is a divide by zero at w = -1, so lets avoid that
    if w == - 1.:
        u = -1.
    else:
        u = (1. + R * w) / sqrt(1. + R * R + 2. * R * w)
    return u


cdef double to_lab_R_lt_1(double R, double w):
    cdef double u, f
    if w < -R:
        # Avoid non-physical results for w=[-1, -R]
        # Do this by assuming a linear shape to u(w)
        # within [-1, -R)
        u = sqrt(1. - R * R)
        f = (w + 1.) / (-R - 1.)
        u = f * (u + 1.) - 1.
    else:
        u = (1. + R * w) / sqrt(1. + R * R + 2. * R * w)

    return u


cdef class TwoBody_TAR:
    """ Class to contain the data and methods for a Target-at-Rest Two-Body
    distribution.
    """

    def __init__(self, adist, Ein, awr, q_value):
        self.adist = adist
        self.R = awr * sqrt((1. + q_value * (awr + 1.) / (awr * Ein)))
        self.awr = awr
        self.q_value = q_value

        if self.R > 1.:
            self.to_lab = to_lab
        elif self.R < 1.:
            self.to_lab = to_lab_R_lt_1
        else:
            self.to_lab = to_lab_R_is_1

    cdef double eval(self, double w, int l):
        return eval_legendre(l, self.to_lab(self.R, w)) * self.adist._eval(w)

    cpdef integrate_cm_legendre(self, double Ein, double[::1] Eouts, int order):
        """This function performs the group-wise integration of a two-body
        kinematic event assuming the target atom is at rest. This function is
        specific to the case of Legendre expansions
        """

        cdef double onep_awr2, Ein_1pR2, inv_2REin, wlo, whi
        cdef int g
        cdef double dmu, u, value, mu_lo, mu_hi
        cdef int l, p
        cdef np.ndarray[np.double_t, ndim=2] integral

        integral = np.zeros((Eouts.shape[0] - 1, order + 1))

        # Calculate the reduced mass and other shorthand parameters
        Ein_1pR2 = Ein * (1. + self.R * self.R)
        onep_awr2 = (1. + self.awr) * (1. + self.awr)
        inv_2REin = 0.5 / (self.R * Ein)

        for g in range(Eouts.shape[0] - 1):
            wlo = (Eouts[g] * onep_awr2 - Ein_1pR2) * inv_2REin
            whi = (Eouts[g + 1] * onep_awr2 - Ein_1pR2) * inv_2REin

            # Adjust to make sure we are in the correct bounds while also
            # skipping the groups we dont need to consider because the
            # energies are too low, and exit if the energies are too
            # high (since we finished all groups)
            if wlo < -1.:
                wlo = -1.
            elif wlo > 1.:
                break
            if whi < -1.:
                continue
            elif whi > 1.:
                whi = 1.

            dmu = whi - wlo
            for l in range(integral.shape[1]):
                integral[g, l] = 0.
                for p in range(_N_QUAD):
                    u = _POINTS[p] * dmu + wlo
                    integral[g, l] += _WEIGHTS[p] * self.eval(u, l)
                integral[g, l] *= dmu

        return integral

    cpdef integrate_cm_histogram(self, double Ein, double[::1] Eouts,
                                 double[::1] mus):
        """This function performs the group-wise integration of a two-body
        kinematic event assuming the target atom is at rest. This function is
        specific to histogram integration.
        """

        cdef double onep_awr2, Ein_1pR2, inv_2REin, wlo, whi
        cdef int g
        cdef double dmu, u, value, mu_lo, mu_hi
        cdef int l, p
        cdef np.ndarray[np.double_t, ndim=2] integral

        integral = np.zeros((Eouts.shape[0] - 1, mus.shape[0] - 1))

        # Calculate the reduced mass and other shorthand parameters
        Ein_1pR2 = Ein * (1. + self.R * self.R)
        onep_awr2 = (1. + self.awr) * (1. + self.awr)
        inv_2REin = 0.5 / (self.R * Ein)

        for g in range(Eouts.shape[0] - 1):
            wlo = (Eouts[g] * onep_awr2 - Ein_1pR2) * inv_2REin
            whi = (Eouts[g + 1] * onep_awr2 - Ein_1pR2) * inv_2REin

            # Adjust to make sure we are in the correct bounds
            if wlo < -1.:
                wlo = -1.
            elif wlo > 1.:
                wlo = 1.
            if whi < -1.:
                whi = -1.
            elif whi > 1.:
                whi = 1.

            # Skip the groups we dont need to consider because the
            # energies are too low, and exit if the energies are too
            # high (since we finished all groups)
            if wlo == whi:
                if wlo == -1.:
                    continue
                elif wlo == 1.:
                    break

            for l in range(mus.shape[0] - 1):
                value = 0.
                mu_lo = max(mus[l], wlo)
                mu_hi = min(mus[l + 1], whi)

                dmu = mu_hi - mu_lo
                for p in range(_N_QUAD):
                    u = _POINTS[p] * dmu + mu_lo
                    value += _WEIGHTS[p] * self.adist._eval(u)
                integral[g, l] = value * dmu

            return integral
