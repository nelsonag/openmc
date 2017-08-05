#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: fast_gil=True

from libc.math cimport sqrt

from scipy.special.cython_special cimport eval_legendre
from scipy.special import roots_sh_legendre
cimport numpy as np
import numpy as np

from angle_distributions cimport *
from .nbody cimport NBody

# The quadrature points can be gathered now, we will get them between 0 and 1
# so the shifting to the proper Elo, Ehi bounds is easier later
cdef int _N_QUAD = 20
cdef double[::1] _POINTS
cdef double[::1] _WEIGHTS
_POINTS, _WEIGHTS = roots_sh_legendre(_N_QUAD)
cdef double[::1] _FMU = np.empty(_N_QUAD)

# The number of outgoing energy points to use when integrating the free-gas
# kernel; we are using simpson's 3/8 rule so this needs to be a multiple of 3
cdef int _N_EOUT = 201
cdef double _N_EOUT_DOUBLE = <double> _N_EOUT

# Set our simpson 3/8 rule coefficients
cdef double[::1] _SIMPSON_WEIGHTS = np.empty(_N_EOUT)
cdef int index
for index in range(_N_EOUT):
    if (index == 0) or (index == _N_EOUT - 1):
        _SIMPSON_WEIGHTS[index] = 0.375
    elif index % 3 == 0:
        _SIMPSON_WEIGHTS[index] = 0.375 * 2.
    else:
        _SIMPSON_WEIGHTS[index] = 0.375 * 3.

# Minimum value of c, the constant used when converting inelastic CM
# distributions to the lab-frame
cdef double _MIN_C = 25.
cdef double _MIN_C2 = _MIN_C * _MIN_C

###############################################################################
# GENERIC QUADRATURE INTEGRATION
###############################################################################


cdef fixed_quad_histogram(func, double[::1] mus, tuple args,
                          double[::1] integral, double min_mu, double max_mu):
    """This function integrates a function between the min_mu and max_mu
    parameters, but subdivided into each of the bins inside the mus
    parameter. The function must have a signature of
    func(x, remaining arguments).
    """
    cdef double dmu, value, u, mu_lo, mu_hi
    cdef int l, p
    for l in range(len(mus) - 1):
        value = 0.
        mu_lo, mu_hi = np.clip(mus[l: l + 2], min_mu, max_mu)
        dmu = mu_hi - mu_lo
        for p in range(_N_QUAD):
            u = _POINTS[p] * dmu + mu_lo
            value += _WEIGHTS[p] * func(u, *args)
        integral[l] = value * dmu


cdef fixed_quad_all_legendres(func, tuple args, double[::1] integral,
                              double[::1] mus_grid, double[:, ::1] wgts):
    """This function integrates a function between -1 and 1 for
    each Legendre order. Since the integration range is [-1,1], a special
    quadrature can be used which incorporates the Legendre values evaluated
    at the points of interest already; simplifying the number of computation.
    The function must have a signature of
    func(x, legendre order, remaining arguments).
    """
    cdef double value, u
    cdef int p, l

    for p in range(len(mus_grid)):
        value = func(mus_grid[p], *args)
        for l in range(integral.shape[0]):
            integral[l] = value * wgts[p, l]


###############################################################################
# FREE-GAS KERNEL INTEGRATION
###############################################################################


cpdef integrate_legendres_freegas(func, double Ein, double[::1] Eouts,
                                  tuple args, double[::1] grid,
                                  double[:, ::1] integral,
                                  double[::1] mus_grid, double[:, ::1] wgts,
                                  double Eout_min, double Eout_max):
    """This function integrates the free-gas kernel across all groups and
    Legendre orders.
    """

    cdef int g, eo, l
    cdef double Eout_lo, Eout_hi, dE, Eout
    cdef tuple eo_args

    for g in range(len(Eouts) - 1):
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
            eo_args = (Eout,) + args
            fixed_quad_all_legendres_freegas(func, eo_args, Ein == Eout, grid,
                                             _FMU, mus_grid, wgts)
            for l in range(integral.shape[1]):
                integral[g, l] += _SIMPSON_WEIGHTS[eo] * grid[l]
        for l in range(integral.shape[1]):
            integral[g, l] *= dE


cdef fixed_quad_all_legendres_freegas(func, tuple args, bint Ein_is_Eout,
                                      double[::1] integral, double [::1] fmu,
                                      double[::1] mus_grid,
                                      double[:, ::1] wgts):
    """This function integrates the free-gas function between -1 and 1 for
    each Legendre order. This method is similar to fixed_quad_all_legendres,
    except it is specialized to deal with the extra factor included by the
    result of func when Eout = Ein
    """
    cdef int p, l
    func(mus_grid, *args, fmu)
    for l in range(integral.shape[0]):
        integral[l] = 0.
    if Ein_is_Eout:
        for p in range(len(mus_grid)):
            if mus_grid[p] < 0.:
                continue
            for l in range(integral.shape[0]):
                if mus_grid[p] == 0.:
                    integral[l] += fmu[p] * wgts[p, l] * sqrt(2.)
                else:
                    integral[l] += fmu[p] * wgts[p, l] * 2. * sqrt(2.)
    else:
        for p in range(len(mus_grid)):
            for l in range(integral.shape[0]):
                integral[l] += fmu[p] * wgts[p, l]


cpdef integrate_histogram_freegas(func, double Ein, double[::1] Eouts,
                                  double[::1] mus, tuple args, double[::1] grid,
                                  double[:, ::1] integral, double Eout_min,
                                  double Eout_max):
    """This function integrates the free-gas kernel across all groups and
    histogram bins.
    """

    cdef int g, eo, l
    cdef double Eout_lo, Eout_hi, dE, Eout
    cdef tuple eo_args

    for g in range(len(Eouts) - 1):
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
            eo_args = (Eout,) + args
            fixed_quad_histogram_freegas(func, mus, eo_args, Ein == Eout, grid,
                                         _FMU)
            for l in range(integral.shape[1]):
                integral[g, l] += _SIMPSON_WEIGHTS[eo] * grid[l]
        for l in range(integral.shape[1]):
            integral[g, l] *= dE


cdef fixed_quad_histogram_freegas(func, double[::1] mus, tuple args,
                                  bint Ein_is_Eout, double[::1] integral,
                                  double [::1] fmu):
    """This function integrates the free-gas function between -1 and 1 for
    each histogram bin. This method is similar to fixed_quad_histogram,
    except it is specialized to deal with the extra factor included by the
    result of func when Eout = Ein
    """
    cdef double dmu, value, u, mu_lo, mu_hi
    cdef int l, p
    for l in range(len(mus) - 1):
        if mus[l + 1] <= 0.:
            continue
        value = 0.
        mu_lo = mus[l]
        mu_hi = mus[l + 1]
        dmu = mu_hi - mu_lo
        for p in range(_N_QUAD):
            _FMU[p] = _POINTS[p] * dmu + mu_lo
        func(_FMU, *args, fmu)
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
    cdef double dmu, u
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

    for g in range(len(Eouts) - 1):
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
            fixed_quad_histogram(adist._eval, mus, (), integral[g, :], wlo, whi)


###############################################################################
# GENERIC CENTER-OF-MASS INTEGRATION METHODS
###############################################################################


cdef inline double integrand_cm(double mu_l, double Eo_l, double c, double Ein,
                                object f, object g, tuple g_args):
    cdef double J, Eo_cm, mu_cm
    J = 1. / sqrt(1. + c * c - 2. * c * mu_l)
    Eo_cm = Eo_l / (J * J)
    mu_cm = J * (mu_l - c)
    return f(Eo_cm) * g(mu_cm, Eo_cm, *g_args)


cpdef integrate_cm(double Ein, double[::1] Eouts, double Eout_cm_max,
                   double awr, double[:, ::1] integral, object edist,
                   object adist, tuple adist_args, bint legendre,
                   double[::1] mus):
    cdef int g, eo, l, orders
    cdef double inv_awrp1, Eout_l_max, Eout_l_min, a, b
    cdef double c, mu_l_min, dE, dmu, u, yg, f, Eout, Eout_prev
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
                _FMU[p] = _WEIGHTS[p] * \
                    integrand_cm(u, Eout, c, Ein, edist, adist, adist_args)

            for l in range(orders):
                y[l] = 0.
                for p in range(_N_QUAD):
                    u = _POINTS[p] * dmu + mu_l_min
                    y[l] += _FMU[p] * eval_legendre(l, u)
                y[l] *= dmu
        else:
            fixed_quad_histogram(integrand_cm, mus,
                                 (Eout, c, Ein, edist, adist, adist_args),
                                 y[:], mu_l_min, 1.)

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



###############################################################################
# LAB-FRAME INTEGRATION METHODS
###############################################################################

# UNCORRELATED
# ------------

cpdef integrate_uncorr_lab(double Ein, double[::1] Eouts,
                           double[:, ::1] integral, double[::1] grid,
                           double[::1] mus_grid, double[:, ::1] wgts,
                           object edist, object adist, tuple adist_args,
                           bint legendre, double[::1] mus):
    cdef int g, eo, l
    cdef double Eout_hi, Eout_lo, Eo_min, Eo_max
    cdef double mu_l_min, dE, Eout, dmu, u
    cdef double[::1] angle_integral

    Eo_min, Eo_max = edist.get_domain(Ein)
    angle_integral = np.empty_like(grid)

    if legendre:
        fixed_quad_all_legendres(adist._eval, (), angle_integral, mus_grid,
                                 wgts)
    else:
        fixed_quad_histogram(adist._eval, mus, (), angle_integral, -1., 1.)

    for g in range(len(Eouts) - 1):
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

        for l in range(grid.shape[0]):
            integral[g, l] = angle_integral[l] * \
                edist.integrate(Eout_lo, Eout_hi)


# CORRELATED
# ----------


cdef double integrand_corr_lab(double mu, double Eout, double Ein,
                                object edist, object adist, tuple adist_args):
    # The integrand used for the lab-frame correlated distributions; this
    # simply combines the energy and angular distributions
    return edist(Eout) * adist(mu, Eout, *adist_args)


cpdef integrate_corr_lab(double Ein, double[::1] Eouts,
                         double[:, ::1] integral, double[::1] grid,
                         double[::1] mus_grid, double[:, ::1] wgts,
                         object edist, object adist, tuple adist_args,
                         bint legendre, double[::1] mus):
    cdef int g, eo, l
    cdef double Eout_hi, Eout_lo, Eo_min, Eo_max
    cdef double mu_l_min, dE, Eout, dmu, u

    Eo_min, Eo_max = edist.get_domain(Ein)

    for g in range(len(Eouts) - 1):
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

            if legendre:
                fixed_quad_all_legendres(integrand_corr_lab,
                                         (Eout, Ein, edist, adist, adist_args),
                                         grid, mus_grid, wgts)
            else:
                fixed_quad_histogram(integrand_corr_lab, mus,
                                     (Eout, Ein, edist, adist, adist_args),
                                     grid, -1., 1.)

            for l in range(integral.shape[1]):
                integral[g, l] += _SIMPSON_WEIGHTS[eo] * grid[l]
        for l in range(integral.shape[1]):
            integral[g, l] *= dE


# NBODY
# -----


cpdef integrate_nbody_lab(double Ein, double[::1] Eouts,
                          double[:, ::1] integral, double[::1] grid,
                          NBody this, bint legendre, double[::1] mus):
    cdef int g, eo, l
    cdef double Eout_hi, Eout_lo, Eo_min, Eo_max
    cdef double mu_l_min, dE, Eout, dmu, u

    Eo_min, Eo_max = this.get_domain(Ein)

    for g in range(len(Eouts) - 1):
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
            mu_l_min = this.mu_min(Eout)

            # Make sure we stay in the allowed bounds
            if mu_l_min < -1.:
                mu_l_min = -1.
            elif mu_l_min > 1.:
                break

            if legendre:
                dmu = 1. - mu_l_min
                for p in range(_N_QUAD):
                    u = _POINTS[p] * dmu + mu_l_min
                    _FMU[p] = _WEIGHTS[p] * this.integrand_legendre_lab(u, Eout)

                for l in range(integral.shape[1]):
                    grid[l] = 0.
                    for p in range(_N_QUAD):
                        u = _POINTS[p] * dmu + mu_l_min
                        grid[l] += _FMU[p] * eval_legendre(l, u)
                    grid[l] *= dmu
            else:
                this.integrand_histogram_lab(Eout, mus, grid)

            for l in range(integral.shape[1]):
                integral[g, l] += _SIMPSON_WEIGHTS[eo] * grid[l]
        for l in range(integral.shape[1]):
            integral[g, l] *= dE
