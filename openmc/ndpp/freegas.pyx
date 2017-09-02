#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False

from libc.math cimport sqrt, exp

from scipy.special.cython_special cimport i0e
from scipy.special import roots_sh_legendre
cimport numpy as np
import numpy as np

from openmc.stats.bisect cimport bisect
from .cevals cimport *
from .cconstants cimport HISTOGRAM, LINLIN, LINLOG, LOGLIN, LOGLOG

"""
The Free-Gas Kernel method used in this code is from
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


cdef double _integrand(double Er, double Eout, double Estar, double E0,
                       double alpha, double most_of_eta, double most_of_mu_cm,
                       double[::1] adist_x, double[::1] adist_p,
                       int adist_interp):
    cdef double mu_cm, eta, value

    mu_cm = 1. - most_of_mu_cm / Er
    eta = most_of_eta * sqrt(Er / Estar - 1.)

    value = exp(-alpha * (Er - E0) + eta) * i0e(eta) * \
        tabular_eval(adist_x, adist_p, adist_x.shape[0] - 1,  adist_interp,
                     mu_cm)

    return value


cdef void calc_Er_integral_cxs(double[::1] mus, double Eout, double Ein,
                               double beta, double alpha, double awr, double kT,
                               double half_beta_2, double[::1] adist_x,
                               double[::1] adist_p, int adist_interp,
                               double[::1] xs_x, double[::1] xs_y,
                               long[::1] xs_bpts, long[::1] xs_interp,
                               double[::1] results):

    cdef double root_in_out, E0, Estar, most_of_eta, b, xc, minE, maxE, value
    cdef double inv_kappa, G, most_of_mu_cm, dE, flo, fhi, mu_l
    cdef int i, u

    root_in_out = sqrt(Ein * Eout)
    for u in range(mus.shape[0]):
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
                         most_of_mu_cm, adist_x, adist_p, adist_interp)
        for i in range(_N_EOUTS_CXS - 1):
            flo = fhi
            fhi = _integrand(minE + float(i + 1) * dE, Eout, Estar, E0, alpha,
                             most_of_eta, most_of_mu_cm, adist_x, adist_p,
                             adist_interp)
            value += flo + fhi
        value *= (maxE - minE) * 0.5

        if Ein != Eout:
            inv_kappa = 0.5 * beta / sqrt(2. * Estar)
        else:
            inv_kappa = 0.5 / sqrt(Ein)

        G = (beta / kT) ** 1.5 * awr * \
            sqrt((awr + 1.) * Eout / (Ein * 2. * _PI)) * inv_kappa

        results[u] = G * value


cdef void calc_Er_integral_doppler(double[::1] mus, double Eout, double Ein,
                                   double beta, double alpha, double awr,
                                   double kT, double half_beta_2,
                                   double[::1] adist_x, double[::1] adist_p,
                                   int adist_interp, double[::1] xs_x,
                                   double[::1] xs_y, long[::1] xs_bpts,
                                   long[::1] xs_interp, double[::1] results):

    cdef double root_in_out, E0, Estar, most_of_eta, b, xc, minE, maxE, value
    cdef double inv_kappa, G, most_of_mu_cm, Epoint, mu_l
    cdef long i, start, end, s, p, k, u
    cdef long interp_type
    cdef double Elo, Ehi, xslo, xshi, dE, Esub_lo, Esub_hi, temp_value, xs_Ein

    root_in_out = sqrt(Ein * Eout)
    for u in range(mus.shape[0]):
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
        if minE > xs_x[0]:
            start = bisect(xs_x, minE) - 1
        else:
            start = 0
        if maxE > xs_x[0]:
            end = bisect(xs_x, maxE, lo=start) - 1
        else:
            end = 0

        xs_Ein = tabulated1d_eval(xs_x, xs_y, xs_x.shape[0] - 1, xs_bpts,
                                  xs_interp, Ein)
        interp_type = LINLIN

        value = 0.
        k = 0
        for i in range(start, end + 1):
            if i == start:
                Elo = minE
                xslo = fast_tabulated1d_eval(Elo, xs_x[i], xs_x[i + 1],
                                             xs_y[i], xs_y[i + 1], interp_type)
            else:
                Elo = xs_x[i]
                xslo = xs_y[i]
            if i == end:
                Ehi = maxE
                xshi = fast_tabulated1d_eval(Ehi, xs_x[i], xs_x[i + 1],
                                             xs_y[i], xs_y[i + 1], interp_type)
            else:
                Ehi = xs_x[i + 1]
                xshi = xs_y[i + 1]

            # Evaluate each of the subdivisions
            dE = (Ehi - Elo) / float(_N_SUBDIVISIONS)
            for s in range(_N_SUBDIVISIONS):
                Esub_lo = Elo + float(s) * dE
                temp_value = 0.
                for p in range(_N_QUAD_ORDER):
                    Epoint = _POINTS[p] * dE + Esub_lo
                    temp_value += _WEIGHTS[p] * \
                        _integrand(Epoint, Eout, Estar, E0, alpha, most_of_eta,
                                   most_of_mu_cm, adist_x, adist_p,
                                   adist_interp) * \
                        fast_tabulated1d_eval(Epoint, Elo, Ehi, xslo, xshi,
                                              interp_type)

                value += temp_value * (Ehi - Elo)

        if Ein != Eout:
            inv_kappa = 0.5 * beta / sqrt(2. * Estar)
        else:
            inv_kappa = 0.5 / sqrt(Ein)

        G = (beta / kT) ** 1.5 * awr * inv_kappa * \
            sqrt((awr + 1.) * Eout / (Ein * 2. * _PI)) / xs_Ein

        results[u] = G * value
