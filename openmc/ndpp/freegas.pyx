#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False

from libc.math cimport sqrt, exp, log, abs

from scipy.special.cython_special cimport i0e, eval_legendre
from scipy.special import roots_sh_legendre
cimport numpy as np
import numpy as np

from openmc.stats.bisect cimport bisect
from openmc.data.function_methods_cython cimport tabulated1d_eval

"""
# The Free-Gas Kernel method used in this code is from
Sanchez, Richard, Alberto Previti, and Domiziano Mostacci.
"New derivation of Blackshow–Murrays formula for the Doppler-broadened
scattering kernel and calculation of the angular moments via Lagrange
interpolation." Annals of Nuclear Energy 87 (2016): 113-125.
"""


cdef double _X_EXP = 30.
cdef double _PI = np.pi

cdef int _N_SUBDIVISIONS = 5
cdef int _N_QUAD_ORDER = 2
cdef int _N_EOUTS_CXS = 200

# The quadrature points can be gathered now, we will get them between 0 and 1
# so the shifting to the proper Elo, Ehi bounds is easier later
cdef double[::1] _POINTS
cdef double[::1] _WEIGHTS
_POINTS, _WEIGHTS = roots_sh_legendre(_N_QUAD_ORDER)


cdef inline double _fast_tabulated1d(double x, double xi, double xi1, double yi,
                                     double yi1, int interp_type):
    """Evaluate a Tabulated1D function when you already know the location in
    the grid and the interpolation type
    """
    cdef double y
    if interp_type == 1:
        # Histogram
        y = yi
    elif interp_type == 2:
        # Linear-linear
        y = yi + (x - xi) / (xi1 - xi) * (yi1 - yi)
    elif interp_type == 3:
        # Linear-log
        y = yi + log(x / xi) / log(xi1 / xi) * (yi1 - yi)
    elif interp_type == 4:
        # Log-linear
        y = yi * exp((x - xi) / (xi1 - xi) * log(yi1 / yi))
    elif interp_type == 5:
        # Log-log
        y = (yi * exp(log(x / xi) / log(xi1 / xi) * log(yi1 / yi)))
    return y


cdef inline double _integrand(double Er, double Eout, double Estar, double E0,
                             double alpha, double most_of_eta,
                             double most_of_mu_cm, object adist):
    cdef double mu_cm, eta, value
    mu_cm = 1. - most_of_mu_cm / Er
    eta = most_of_eta * sqrt(Er / Estar - 1.)

    value = exp(-alpha * (Er - E0) + eta) * adist._eval(mu_cm) * i0e(eta)

    return value


cpdef calc_Er_integral_cxs(double[::1] mus, double Eout, double Ein,
                           double beta, double alpha, double awr, double kT,
                           double half_beta_2, object adist, object xs,
                           double[::1] results):
    cdef double root_in_out, E0, Estar, most_of_eta, b, xc, minE, maxE, value
    cdef double inv_kappa, G, most_of_mu_cm, dE, flo, fhi, mu_l
    cdef int i, u

    root_in_out = sqrt(Ein * Eout)
    for u in range(len(mus)):
        mu_l = mus[u]
        E0 = Ein / awr - beta * root_in_out * mu_l
        Estar = half_beta_2 * (Ein + Eout - 2. * root_in_out * mu_l)
        most_of_eta = (awr + 1.) / kT * root_in_out * sqrt(1. - mu_l * mu_l)
        most_of_mu_cm = 0.5 * beta * beta * (Ein + Eout - 2. *
                                             root_in_out * mu_l)
        b = 0.5 * beta * root_in_out * sqrt(alpha * (1. - mu_l * mu_l) / Estar)
        xc = _X_EXP + alpha * (E0 - Estar) + b * b
        if xc < 0.:
            results[u] = 0.
            continue
        xc = sqrt(xc)
        minE = Estar + max(0., b - xc)**2 / alpha
        maxE = Estar + (b + xc)**2 / alpha
        dE = (maxE - minE) / float(_N_EOUTS_CXS - 1)

        # Perform trapezoidal integration over all the Eout points
        value = 0.
        fhi = _integrand(minE, Eout, Estar, E0, alpha, most_of_eta,
                         most_of_mu_cm, adist)
        for i in range(_N_EOUTS_CXS - 1):
            flo = fhi
            fhi = _integrand(minE + float(i + 1) * dE, Eout, Estar, E0, alpha,
                             most_of_eta, most_of_mu_cm, adist)
            value += flo + fhi
        value *= (maxE - minE) * 0.5

        if Ein != Eout:
            inv_kappa = 0.5 * beta / sqrt(2. * Estar)
        else:
            inv_kappa = 0.5 / sqrt(Ein)

        G = (beta / kT) ** 1.5 * awr * \
            sqrt((awr + 1.) * Eout / (Ein * 2. * _PI)) * inv_kappa

        results[u] = G * value


cpdef calc_Er_integral_doppler(double[::1] mus, double Eout, double Ein,
                               double beta, double alpha, double awr,
                               double kT, double half_beta_2, object adist,
                               object xs, double[::1] results):
    cdef double root_in_out, E0, Estar, most_of_eta, b, xc, minE, maxE, value
    cdef double inv_kappa, G, most_of_mu_cm, Epoint, mu_l
    cdef int i, start, end, s, p, k, u
    cdef int interp_type
    cdef double Elo, Ehi, xslo, xshi, dE, Esub_lo, Esub_hi, temp_value, xs_Ein

    root_in_out = sqrt(Ein * Eout)
    for u in range(len(mus)):
        mu_l = mus[u]
        E0 = Ein / awr - beta * root_in_out * mu_l
        Estar = half_beta_2 * (Ein + Eout - 2. * root_in_out * mu_l)
        most_of_eta = (awr + 1.) / kT * root_in_out * sqrt(1. - mu_l * mu_l)
        most_of_mu_cm = 0.5 * beta * beta * (Ein + Eout - 2. *
                                             root_in_out * mu_l)
        b = 0.5 * beta * root_in_out * sqrt(alpha * (1. - mu_l * mu_l) / Estar)
        xc = _X_EXP + alpha * (E0 - Estar) + b * b
        if xc < 0.:
            results[u] = 0.
            continue
        xc = sqrt(xc)
        minE = Estar + max(0., b - xc)**2 / alpha
        maxE = Estar + (b + xc)**2 / alpha

        # Find the locations of minE and maxE in the cross section grid
        if minE > xs._x[0]:
            start = bisect(xs._x, minE) - 1
        else:
            start = 0
        if maxE > xs._x[0]:
            end = bisect(xs._x, maxE, lo=start) - 1
        else:
            end = 0

        xs_Ein = tabulated1d_eval(xs, Ein)

        value = 0.
        k = 0
        for i in range(start, end + 1):
            if i == start:
                Elo = minE
                xslo = tabulated1d_eval(xs, Elo)
            else:
                Elo = xs._x[i]
                xslo = xs._y[i]
            if i == end:
                Ehi = maxE
                xshi = tabulated1d_eval(xs, Ehi)
            else:
                Ehi = xs._x[i + 1]
                xshi = xs._y[i + 1]
            # get the interpolation type
            if xs._breakpoints.shape[0] == 1:
                k = 0
            else:
                k = bisect(xs._breakpoints, i, lo=k) - 1
            interp_type = xs._interpolation[k]

            # Evaluate each of the subdivisions
            dE = (Ehi - Elo) / float(_N_SUBDIVISIONS)
            for s in range(_N_SUBDIVISIONS):
                Esub_lo = Elo + float(s) * dE
                temp_value = 0.
                for p in range(_N_QUAD_ORDER):
                    Epoint = _POINTS[p] * dE + Esub_lo
                    temp_value += _WEIGHTS[p] * \
                        _integrand(Epoint, Eout, Estar, E0, alpha, most_of_eta,
                                   most_of_mu_cm, adist) * \
                        _fast_tabulated1d(Epoint, Elo, Ehi, xslo, xshi,
                                          interp_type)

                value += temp_value * (Ehi - Elo)

        if Ein != Eout:
            inv_kappa = 0.5 * beta / sqrt(2. * Estar)
        else:
            inv_kappa = 0.5 / sqrt(Ein)

        G = (beta / kT) ** 1.5 * awr * inv_kappa * \
            sqrt((awr + 1.) * Eout / (Ein * 2. * _PI)) / xs_Ein

        results[u] = G * value
