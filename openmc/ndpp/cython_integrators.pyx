#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False

from libc.math cimport sqrt

from scipy.special.cython_special cimport eval_legendre
cimport numpy as np
import numpy as np

from .cconstants cimport *
from .energyangledist cimport EnergyAngle_Cython
from .freegas cimport CALC_ER_INTEGRAL, calc_Er_integral_doppler, \
                      calc_Er_integral_cxs

# Function pointer to use for free-gas integration
cdef CALC_ER_INTEGRAL _FREEGAS_FUNCTION


###############################################################################
# FREE-GAS KERNEL INTEGRATION
###############################################################################


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


cpdef integrate_legendres_freegas(double Ein, double[::1] Eouts, double beta,
                                  double alpha, double awr, double kT,
                                  double half_beta_2, double[::1] adist_x,
                                  double[::1] adist_p, int adist_interp,
                                  double[::1] xs_x, double[::1] xs_y,
                                  long[::1] xs_bpts, long[::1] xs_interp,
                                  double[::1] grid, double[:, ::1] integral,
                                  double[::1] mus_grid, double[:, ::1] wgts,
                                  double Eout_min, double Eout_max):
    """This function integrates the free-gas kernel across all groups and
    Legendre orders.
    """

    cdef int g, eo, l
    cdef double Eout_lo, Eout_hi, dE, Eout

    for g in range(Eouts.shape[0] - 1):
        # If our group is below the possible outgoing energies, just skip it
        if Eouts[g + 1] <= Eout_min:
            continue
        # If our group is above the max energy then we are all done
        if Eouts[g] >= Eout_max:
            break
        Eout_lo = max(Eout_min, Eouts[g])
        Eout_hi = min(Eout_max, Eouts[g + 1])

        dE = (Eout_hi - Eout_lo) / (_N_EOUT_DOUBLE - 1.)
        if Eout_lo == Ein:
            grid[:] = 0.

        for eo in range(_N_EOUT):
            Eout = Eout_lo + ((<double>eo) * dE)
            fixed_quad_all_legendres_freegas(Eout, Ein, beta, alpha, awr, kT,
                                             half_beta_2, adist_x, adist_p,
                                             adist_interp, xs_x, xs_y, xs_bpts,
                                             xs_interp, Ein == Eout, grid,
                                             _FMU, mus_grid, wgts)
            for l in range(integral.shape[1]):
                integral[g, l] += _SIMPSON_WEIGHTS[eo] * grid[l]
        for l in range(integral.shape[1]):
            integral[g, l] *= dE


cdef fixed_quad_all_legendres_freegas(double Eout, double Ein, double beta,
                                      double alpha, double awr, double kT,
                                      double half_beta_2, double[::1] adist_x,
                                      double[::1] adist_p, int adist_interp,
                                      double[::1] xs_x, double[::1] xs_y,
                                      long[::1] xs_bpts, long[::1] xs_interp,
                                      bint Ein_is_Eout, double[::1] integral,
                                      double [::1] fmu, double[::1] mus_grid,
                                      double[:, ::1] wgts):
    """This function integrates a function between -1 and 1 for
    each Legendre order. Since the integration range is [-1,1], a special
    quadrature can be used which incorporates the Legendre values evaluated
    at the points of interest already; simplifying the number of computation.
    This function is specialized to deal with the extra factor included when
    Eout == Ein.
    """

    cdef int p, l

    _FREEGAS_FUNCTION(mus_grid, Eout, Ein, beta, alpha, awr, kT, half_beta_2,
                      adist_x, adist_p, adist_interp, xs_x, xs_y, xs_bpts,
                      xs_interp, fmu)
    for l in range(integral.shape[0]):
        integral[l] = 0.
    if Ein_is_Eout:
        for p in range(mus_grid.shape[0]):
            if mus_grid[p] < 0.:
                continue
            for l in range(integral.shape[0]):
                if mus_grid[p] == 0.:
                    integral[l] += fmu[p] * wgts[p, l] * sqrt(2.)
                else:
                    integral[l] += fmu[p] * wgts[p, l] * 2. * sqrt(2.)
    else:
        for p in range(mus_grid.shape[0]):
            for l in range(integral.shape[0]):
                integral[l] += fmu[p] * wgts[p, l]


cpdef integrate_histogram_freegas(bint use_cxs, double Ein,
                                  double[::1] Eouts, double[::1] mus,
                                  double beta, double alpha, double awr,
                                  double kT, double half_beta_2,
                                  double[::1] adist_x, double[::1] adist_p,
                                  int adist_interp, double[::1] xs_x,
                                  double[::1] xs_y, long[::1] xs_bpts,
                                  long[::1] xs_interp, double[::1] grid,
                                  double[:, ::1] integral, double Eout_min,
                                  double Eout_max):
    """This function integrates the free-gas kernel across all groups and
    histogram bins.
    """

    cdef int g, eo, l
    cdef double Eout_lo, Eout_hi, dE, Eout
    cdef CALC_ER_INTEGRAL func

    if use_cxs:
        func = <CALC_ER_INTEGRAL> calc_Er_integral_cxs
    else:
        func = <CALC_ER_INTEGRAL> calc_Er_integral_doppler

    for g in range(Eouts.shape[0] - 1):
        # If our group is below the possible outgoing energies, just skip it
        if Eouts[g + 1] <= Eout_min:
            continue
        # If our group is above the max energy then we are all done
        if Eouts[g] >= Eout_max:
            break
        Eout_lo = max(Eout_min, Eouts[g])
        Eout_hi = min(Eout_max, Eouts[g + 1])
        dE = (Eout_hi - Eout_lo) / (_N_EOUT_DOUBLE - 1.)
        if Eout_lo == Ein:
            grid[:] = 0.
        for eo in range(_N_EOUT):
            Eout = Eout_lo + ((<double>eo) * dE)
            fixed_quad_histogram_freegas(func, mus, Eout, Ein, beta, alpha, awr,
                                         kT, half_beta_2, adist_x, adist_p,
                                         adist_interp, xs_x, xs_y, xs_bpts,
                                         xs_interp, Ein == Eout, grid, _FMU)
            for l in range(integral.shape[1]):
                integral[g, l] += _SIMPSON_WEIGHTS[eo] * grid[l]
        for l in range(integral.shape[1]):
            integral[g, l] *= dE


cdef fixed_quad_histogram_freegas(CALC_ER_INTEGRAL func, double[::1] mus,
                                  double Eout, double Ein, double beta,
                                  double alpha, double awr, double kT,
                                  double half_beta_2, double[::1] adist_x,
                                  double[::1] adist_p, int adist_interp,
                                  double[::1] xs_x, double[::1] xs_y,
                                  long[::1] xs_bpts, long[::1] xs_interp,
                                  bint Ein_is_Eout, double[::1] integral,
                                  double [::1] fmu):
    """This function integrates a function between the min_mu and max_mu
    parameters, but subdivided into each of the bins inside the mus
    parameter. This method is specialized to deal with the extra factor
    included by the result of the func when Eout = Ein.
    """

    cdef double dmu, value, u, mu_lo, mu_hi
    cdef int l, p

    for l in range(mus.shape[0] - 1):
        if mus[l + 1] <= 0.:
            continue
        value = 0.
        mu_lo = mus[l]
        mu_hi = mus[l + 1]
        dmu = mu_hi - mu_lo
        for p in range(_N_QUAD):
            _FMU[p] = _POINTS[p] * dmu + mu_lo
        func(_FMU, Eout, Ein, beta, alpha, awr, kT, half_beta_2, adist_x,
             adist_p, adist_interp, xs_x, xs_y, xs_bpts, xs_interp, fmu)

        if Ein_is_Eout:
            for p in range(_N_QUAD):
                if Ein_is_Eout and _FMU[p] == 0.:
                    value += _WEIGHTS[p] * fmu[p] * sqrt(2.)
                else:
                    value += _WEIGHTS[p] * fmu[p] * 2. * sqrt(2.)
        else:
            for p in range(_N_QUAD):
                value += _WEIGHTS[p] * fmu[p]
        integral[l] = value * dmu


###############################################################################
# TWO-BODY KINEMATICS INTEGRATION METHODS
###############################################################################


cdef inline double to_lab(double R, double w):
    return (1. + R * w) / sqrt(1. + R * R + 2. * R * w)


cdef inline double to_lab_R_is_1(double R, double w):
    cdef double u
    # There is a divide by zero at w = -1, so lets avoid that
    if w == - 1.:
        u = -1.
    else:
        u = (1. + R * w) / sqrt(1. + R * R + 2. * R * w)
    return u


cdef inline double to_lab_R_lt_1(double R, double w):
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


cdef double integrand_cm_tar_legendre(double w, int l, double R, object adist):
    return eval_legendre(l, to_lab(R, w)) * adist._eval(w)


cdef double integrand_cm_tar_legendre_R_is_1(double w, int l, double R,
                                             object adist):
    return eval_legendre(l, to_lab_R_is_1(R, w)) * adist._eval(w)


cdef double integrand_cm_tar_legendre_R_lt_1(double w, int l, double R,
                                             object adist):
    return eval_legendre(l, to_lab_R_lt_1(R, w)) * adist._eval(w)


ctypedef double (*INTEGRAND_CM_TAR_LEGENDRE)(double w, int l, double R,
                                             object adist)


cpdef integrate_twobody_cm_TAR(double Ein, double[::1] Eouts,
                               double awr, double q_value, object adist,
                               double[:, ::1] integral, bint legendre,
                               double[::1] mus):
    """This function performs the group-wise integration of a two-body
    kinematic event assuming the target atom is at rest
    """
    cdef double R, onep_awr2, Ein_1pR2, inv_2REin, Eo_lo, Eo_hi, wlo, whi
    cdef int g
    cdef INTEGRAND_CM_TAR_LEGENDRE func
    cdef double dmu, u, value, mu_lo, mu_hi
    cdef int l, p

    # Calculate the reduced mass and other shorthand parameters
    R = awr * sqrt((1. + q_value * (awr + 1.) / (awr * Ein)))
    Ein_1pR2 = Ein * (1. + R * R)
    onep_awr2 = (1. + awr) * (1. + awr)
    inv_2REin = 0.5 / (R * Ein)

    # Our to_lab function has some divide by zero pitfalls in it.
    # In order to not penalize 99% of calculations, special functions exist
    # to handle when those scenarios come up
    if legendre:
        if R > 1.:
            func = integrand_cm_tar_legendre
        elif R < 1.:
            func = integrand_cm_tar_legendre_R_lt_1
        else:
            func = integrand_cm_tar_legendre_R_is_1

    for g in range(Eouts.shape[0] - 1):
        Eo_lo = Eouts[g]
        Eo_hi = Eouts[g + 1]
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

        if legendre:
            dmu = whi - wlo
            for l in range(integral.shape[1]):
                integral[g, l] = 0.
                for p in range(_N_QUAD):
                    u = _POINTS[p] * dmu + wlo
                    integral[g, l] += _WEIGHTS[p] * func(u, l, R, adist)
                integral[g, l] *= dmu
        else:
            for l in range(mus.shape[0] - 1):
                value = 0.
                mu_lo = max(mus[l], wlo)
                mu_hi = min(mus[l + 1], whi)

                dmu = mu_hi - mu_lo
                for p in range(_N_QUAD):
                    u = _POINTS[p] * dmu + mu_lo
                    value += _WEIGHTS[p] * adist._eval(u)
                integral[g, l] = value * dmu


###############################################################################
# GENERIC CENTER-OF-MASS INTEGRATION METHODS
###############################################################################


cdef inline double integrand_cm(double mu_l, double Eo_l, double c, double Ein,
                                EnergyAngle_Cython eadist):
    cdef double J, Eo_cm, mu_cm
    J = 1. / sqrt(1. + c * c - 2. * c * mu_l)
    Eo_cm = Eo_l / (J * J)
    mu_cm = J * (mu_l - c)
    return eadist.eval(mu_cm, Eo_cm)


cpdef integrate_cm(double Ein, double[::1] Eouts, double Eout_cm_max,
                   double awr, double[:, ::1] integral,
                   EnergyAngle_Cython eadist,
                   bint legendre, double[::1] mus):
    cdef int g, eo, l, orders, p
    cdef double inv_awrp1, Eout_l_max, Eout_l_min, a, b
    cdef double c, mu_l_min, dE, dmu, u, yg, f, Eout, Eout_prev, value, mu_lo
    cdef double[::1] y
    cdef double[::1] y_prev

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

        if legendre:
            dmu = 1. - mu_l_min
            for p in range(_N_QUAD):
                u = _POINTS[p] * dmu + mu_l_min
                _FMU[p] = _WEIGHTS[p] * integrand_cm(u, Eout, c, Ein, eadist)

            for l in range(orders):
                y[l] = 0.
                for p in range(_N_QUAD):
                    u = _POINTS[p] * dmu + mu_l_min
                    y[l] += _FMU[p] * eval_legendre(l, u)
                y[l] *= dmu
        else:
            for l in range(mus.shape[0] - 1):
                value = 0.
                mu_lo = max(mus[l], mu_l_min)

                dmu = mus[l + 1] - mu_lo
                for p in range(_N_QUAD):
                    u = _POINTS[p] * dmu + mu_lo
                    value += _WEIGHTS[p] * integrand_cm(u, Eout, c, Ein, eadist)
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
                for l in range(orders):
                    yg = (1. - f) * y[l]  + f * y_prev[l]
                    integral[g, l] += a * (y_prev[l] + yg)
                    # And add the top portion to the next group
                    integral[g + 1, l] += b * (yg + y[l])
                # Move our group counter to the next group
                g += 1
            else:
                # Then there is no group straddling, so just do a standard
                # trapezoidal integration
                a = 0.5 * (Eout - Eout_prev)
                for l in range(orders):
                    integral[g, l] += a * (y_prev[l] + y[l])
