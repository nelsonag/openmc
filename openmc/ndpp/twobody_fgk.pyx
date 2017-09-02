#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False

import numpy as np
import scipy.optimize as sopt


# The free-gas kernel (FGK) functions asymptotically go to zero as outgoing
# energy varies for a given value of mu. We dont want to integrate over all
# that space. The following constants are used in identifying a more useful
# range to integrate over

# This term sets the fraction of the peak (for a given mu) of the FGK function
# to look for when finding the lower and upper Eout values of integration
_FGK_ROOT_TOL = 1.e-2

# These are constants we initialize once instead of every time through that are
# used during the search for the FGK roots
# The mu value to use on the back-scatter end to find the roots
_MU_BACK = np.array([-1.])
# The mu value to use on the front-scatter end to find the roots
# We dont use mu = 1. since the function behaves poorly there
_MU_FWD = np.array([0.9])
# Set the mu interpolant to extend the range found at _MU_BACK and _MU_BACK
# all the way to a forward-scatter mu of 1.0
_INTERP_MU = (_MU_FWD[0] - (_MU_BACK[0])) * 0.5
# Store space for our results
_FGK_RESULT = np.empty_like(_MU_BACK)
# Function pointer to use for free-gas integration
cdef CALC_ER_INTEGRAL _FREEGAS_FUNCTION


cdef class TwoBody_FGK:
    """ Class to contain the data and methods for a Target-at-Rest Two-Body
    distribution
    """

    def __init__(self, adist, double Ein, double awr, double kT, xs,
                 double[::1] Eouts):
        # Convert the string interpolation keys to integers
        if adist._interpolation == 'histogram':
            self.adist_interp = 1
        elif adist._interpolation == 'linear-linear':
            self.adist_interp = 2
        elif adist._interpolation == 'linear-log':
            self.adist_interp = 3
        elif adist._interpolation == 'log-linear':
            self.adist_interp = 4
        elif adist._interpolation == 'log-log':
            self.adist_interp = 5
        self.adist_x = adist._x
        self.adist_p = adist._p
        self.xs_x = xs._x
        self.xs_y = xs._y
        self.xs_bpts = xs._breakpoints
        self.xs_interp = xs._interpolation

        # Set the constants to be used in our calculation
        self.beta = (awr + 1.) / awr
        self.half_beta_2 = 0.25 * self.beta * self.beta
        self.alpha = awr / kT
        self.awr = awr
        self.kT = kT

        # self.Eout_min = Eout_min
        # self.Eout_max = Eout_max

        # Set the constants to be used in our calculation
        beta = (awr + 1.) / awr
        half_beta_2 = 0.25 * beta * beta
        alpha = awr / kT
        srch_args = (Ein, self.beta, self.alpha, self.awr, self.kT,
                     self.half_beta_2, adist._x, adist._p, self.adist_interp,
                     xs._x, xs._y, xs._breakpoints, xs._interpolation)

        # Find the Eout bounds
        # We will search both backward and forward scattering conditions to see
        # what the most extreme outgoing energies we need to consider are

        # Do the following for backward scattering
        # Now find the maximum value so we can bracket our range and also use it
        # to find our roots (when function passes tolerance * max)
        maximum = sopt.minimize_scalar(find_freegas_integral_max,
                                       bracket=[0., Ein + 12. / self.alpha],
                                       args=(_MU_BACK, *srch_args, _FGK_RESULT))
        Eout_peak = maximum.x
        func_peak = -maximum.fun

        # Now find the upper and lower roots around this peak
        Eout_min_back = \
            sopt.brentq(find_freegas_integral_root, Eouts[0], Eout_peak,
                        args=(_MU_BACK, *srch_args, _FGK_RESULT,
                              _FGK_ROOT_TOL * func_peak))
        Eout_max_back = \
            sopt.brentq(find_freegas_integral_root, Eout_peak,
                        Eouts[Eouts.shape[0] - 1],
                        args=(_MU_BACK, *srch_args, _FGK_RESULT,
                              _FGK_ROOT_TOL * func_peak))

        # Now we have to repeat for the top end;
        # Here we dont do mu=1, since the problem is unstable at mu=1, Ein=Eout,
        # which would give us a peak many orders of magnitude higher than the
        # surrounding points (so almost any point would pass the tolerance * peak
        # root check even with pretty low tolerances).
        # Instead we search at mu=0.9 and linearly interpolate to mu=1.
        maximum = \
            sopt.minimize_scalar(find_freegas_integral_max,
                                 bracket=[0., Ein + 12. / self.alpha],
                                 args=(_MU_FWD, *srch_args, _FGK_RESULT))
        Eout_peak = maximum.x
        func_peak = -maximum.fun
        Eout_min_fwd = sopt.brentq(find_freegas_integral_root,
                                   Eouts[0], Eout_peak,
                                   args=(_MU_FWD, *srch_args, _FGK_RESULT,
                                         _FGK_ROOT_TOL * func_peak))
        Eout_max_fwd = sopt.brentq(find_freegas_integral_root,
                                   Eout_peak, Eouts[Eouts.shape[0] - 1],
                                   args=(_MU_FWD, *srch_args, _FGK_RESULT,
                                         _FGK_ROOT_TOL * func_peak))

        Eout_min_fwd = (1. - _INTERP_MU) * Eout_min_back + \
            _INTERP_MU * Eout_min_fwd
        Eout_max_fwd = (1. - _INTERP_MU) * Eout_max_back + \
            _INTERP_MU * Eout_max_fwd
        Eout_min = min(Eout_min_back, Eout_min_fwd)
        Eout_max = max(Eout_max_back, Eout_max_fwd)

        self.Eout_min = Eout_min
        self.Eout_max = Eout_max

    cpdef integrate_cm_legendre(self, double Ein, double[::1] Eouts, int order,
                                double[::1] mus_grid, double[:, ::1] wgts):
        """This function integrates the free-gas kernel across all groups and
        Legendre orders.
        """
        cdef int g, eo, l, p
        cdef double Eout_lo, Eout_hi, dE, Eout
        cdef np.ndarray[np.double_t, ndim=2] integral
        cdef np.ndarray[np.double_t, ndim=1] grid

        integral = np.zeros((Eouts.shape[0] - 1, order + 1))
        grid = np.empty(integral.shape[1])

        for g in range(Eouts.shape[0] - 1):
            # If our group is below the possible outgoing energies, just skip it
            if Eouts[g + 1] <= self.Eout_min:
                continue
            # If our group is above the max energy then we are all done
            if Eouts[g] >= self.Eout_max:
                break
            Eout_lo = max(self.Eout_min, Eouts[g])
            Eout_hi = min(self.Eout_max, Eouts[g + 1])

            dE = (Eout_hi - Eout_lo) / (_N_EOUT_DOUBLE - 1.)

            for eo in range(_N_EOUT):
                Eout = Eout_lo + ((<double>eo) * dE)

                _FREEGAS_FUNCTION(mus_grid, Eout, Ein, self.beta,
                                  self.alpha, self.awr, self.kT,
                                  self.half_beta_2, self.adist_x,
                                  self.adist_p, self.adist_interp,
                                  self.xs_x, self.xs_y, self.xs_bpts,
                                  self.xs_interp, _FMU)

                for l in range(grid.shape[0]):
                    grid[l] = 0.

                if Ein == Eout:
                    for p in range(mus_grid.shape[0]):
                        if mus_grid[p] < 0.:
                            continue
                        for l in range(grid.shape[0]):
                            if mus_grid[p] == 0.:
                                grid[l] += _FMU[p] * wgts[p, l] * sqrt(2.)
                            else:
                                grid[l] += _FMU[p] * wgts[p, l] * 2. * sqrt(2.)
                else:
                    for p in range(mus_grid.shape[0]):
                        for l in range(grid.shape[0]):
                            grid[l] += _FMU[p] * wgts[p, l]

                for l in range(integral.shape[1]):
                    integral[g, l] += _SIMPSON_WEIGHTS[eo] * grid[l]
            for l in range(integral.shape[1]):
                integral[g, l] *= dE

        return integral

    cpdef integrate_cm_histogram(self, double Ein, double[::1] Eouts,
                                 double[::1] mus):
        """This function integrates the free-gas kernel across all groups and
        histogram bins.
        """

        cdef int g, eo, l, p
        cdef double Eout_lo, Eout_hi, dE, Eout, dmu, value, u
        cdef np.ndarray[np.double_t, ndim=2] integral
        cdef np.ndarray[np.double_t, ndim=1] grid

        integral = np.zeros((Eouts.shape[0] - 1, mus.shape[0] - 1))
        grid = np.empty(integral.shape[1])

        for g in range(Eouts.shape[0] - 1):
            # If our group is below the possible outgoing energies, just skip it
            if Eouts[g + 1] <= self.Eout_min:
                continue
            # If our group is above the max energy then we are all done
            if Eouts[g] >= self.Eout_max:
                break
            Eout_lo = max(self.Eout_min, Eouts[g])
            Eout_hi = min(self.Eout_max, Eouts[g + 1])
            dE = (Eout_hi - Eout_lo) / (_N_EOUT_DOUBLE - 1.)
            if Eout_lo == Ein:
                grid[:] = 0.
            for eo in range(_N_EOUT):
                Eout = Eout_lo + ((<double>eo) * dE)

                for l in range(mus.shape[0] - 1):
                    if mus[l + 1] <= 0.:
                        continue

                    value = 0.
                    dmu = mus[l + 1] - mus[l]
                    for p in range(_N_QUAD):
                        _MU[p] = _POINTS[p] * dmu + mus[l]

                    _FREEGAS_FUNCTION(_MU, Eout, Ein, self.beta,
                                      self.alpha, self.awr, self.kT,
                                      self.half_beta_2, self.adist_x,
                                      self.adist_p, self.adist_interp,
                                      self.xs_x, self.xs_y, self.xs_bpts,
                                      self.xs_interp, _FMU)

                    if Ein == Eout:
                        for p in range(_N_QUAD):
                            if _MU[p] == 0.:
                                value += _WEIGHTS[p] * _FMU[p] * sqrt(2.)
                            else:
                                value += _WEIGHTS[p] * _FMU[p] * 2. * sqrt(2.)
                    else:
                        for p in range(_N_QUAD):
                            value += _WEIGHTS[p] * _FMU[p]
                    grid[l] = value * dmu

                for l in range(integral.shape[1]):
                    integral[g, l] += _SIMPSON_WEIGHTS[eo] * grid[l]
            for l in range(integral.shape[1]):
                integral[g, l] *= dE

        return integral


def set_freegas_method(bint use_cxs):
    # This method sets the free-gas function to use, depending on if using the
    # constant x/s approximation or incorporating thermal motion
    global _FREEGAS_FUNCTION
    if use_cxs:
        _FREEGAS_FUNCTION = calc_Er_integral_cxs
    else:
        _FREEGAS_FUNCTION = calc_Er_integral_doppler


def find_freegas_integral_max(double Eout, double[::1] mu, double Ein,
                              double beta, double alpha, double awr, double kT,
                              double half_beta_2, double[::1] adist_x,
                              double[::1] adist_p, int adist_interp,
                              double[::1] xs_x, double[::1] xs_y,
                              long[::1] xs_bpts, long[::1] xs_interp,
                              double[::1] result):
    # Function used to find the maximum value of the FGK(Eout) at a given mu
    _FREEGAS_FUNCTION(mu, Eout, Ein, beta, alpha, awr, kT, half_beta_2,
                      adist_x, adist_p, adist_interp, xs_x, xs_y, xs_bpts,
                      xs_interp, result)
    return -result[0]


def find_freegas_integral_root(double Eout, double[::1] mu, double Ein,
                               double beta, double alpha, double awr, double kT,
                               double half_beta_2, double[::1] adist_x,
                               double[::1] adist_p, int adist_interp,
                               double[::1] xs_x, double[::1] xs_y,
                               long[::1] xs_bpts, long[::1] xs_interp,
                               double[::1]result, double value):
    # Function used to find the root of the FGK(Eout) at a given mu
    _FREEGAS_FUNCTION(mu, Eout, Ein, beta, alpha, awr, kT, half_beta_2,
                      adist_x, adist_p, adist_interp, xs_x, xs_y, xs_bpts,
                      xs_interp, result)
    return result[0] - value