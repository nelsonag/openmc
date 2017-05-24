#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False

from libc.math cimport sqrt, sinh, cosh

from scipy.special.cython_special cimport eval_legendre
from scipy.special import roots_sh_legendre
cimport numpy as np
import numpy as np

from openmc.stats.bisect cimport bisect
# from cython_classes cimport NBody_CM_Cython
from angle_distributions cimport *

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
cdef double _N_EOUT_DOUBLE = 201.

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
                   double awr, double[::1] grid, double[:, ::1] integral,
                   object edist, object adist, tuple adist_args,
                   bint legendre, double[::1] mus):
    cdef int g, eo, l
    cdef double root_awrp12, Eout_l_max, Eout_l_min, Eout_hi, Eout_lo
    cdef double c, mu_l_min, dE, Eout, dmu, u

    root_awrp12 = sqrt(1. / ((awr + 1.) * (awr + 1.)))
    Eout_l_max = Ein * (sqrt(Eout_cm_max / Ein) + root_awrp12)**2
    Eout_l_min = Ein * (sqrt(Eout_cm_max / Ein) - root_awrp12)**2

    for g in range(len(Eouts) - 1):
        Eout_lo = Eouts[g]
        Eout_hi = Eouts[g + 1]

        # If our group is below the possible outgoing energies, just skip it
        if Eout_hi < Eout_l_min:
            continue
        # If our group is above the max energy then we are all done
        if Eout_lo > Eout_l_max:
            break

        Eout_lo = max(Eout_l_min, Eout_lo)
        Eout_hi = min(Eout_l_max, Eout_hi)

        dE = (Eout_hi - Eout_lo) / (_N_EOUT_DOUBLE - 1.)

        for eo in range(_N_EOUT):
            Eout = Eout_lo + ((<double>eo) * dE)
            c = root_awrp12 * sqrt(Ein / Eout)
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

                for l in range(integral.shape[1]):
                    grid[l] = 0.
                    for p in range(_N_QUAD):
                        u = _POINTS[p] * dmu + mu_l_min
                        grid[l] += _FMU[p] * eval_legendre(l, u)
                    grid[l] *= dmu
            else:
                fixed_quad_histogram(integrand_cm, mus,
                                     (Eout, c, Ein, edist, adist, adist_args),
                                     grid, mu_l_min, 1.)

            for l in range(integral.shape[1]):
                integral[g, l] += _SIMPSON_WEIGHTS[eo] * grid[l]
        for l in range(integral.shape[1]):
            integral[g, l] *= dE


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
# ------------


cdef class NBody:
    """ Class to contain the data and methods for a Nbody reaction; this is
    the only distribution type with a class because it does not have a
    pre-initialized edist and adist function available.
    """
    cdef public double Emax, Estar, C, exponent

    def __init__(self, this, Ein):
        # Find the Emax, C, and exponent parameters
        Ap = this.total_mass
        self.Emax = (Ap - 1.) / Ap * \
            (this.atomic_weight_ratio / (this.atomic_weight_ratio + 1.) * \
             Ein + this.q_value)
        # Estar is also ECM in some of the ENDF/NJOY references
        self.Estar = Ein * (1. / (this.atomic_weight_ratio + 1.))
        if this.n_particles == 3:
            self.C = 4. / (np.pi * (self.Emax * self.Emax))
            self.exponent = 0.5
        elif this.n_particles == 4:
            self.C = 105. / (32. * np.sqrt(self.Emax**7))
            self.exponent = 2
        elif this.n_particles == 5:
            self.C = 256. / (14. * np.pi * (self.Emax**5))
            self.exponent = 3.5

    cpdef get_domain(self, double Ein=0.):
        return 0., self.Emax

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
