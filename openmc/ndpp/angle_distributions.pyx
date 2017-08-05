#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False


from libc.math cimport sqrt, sinh, cosh

from openmc.stats.bisect cimport bisect


"""This module contains the angular distribution functions used to wrap the
energy and angle distributions. Due to the speed benefits and lack of a need
for support of arbitrary distribution types, the Kalbach-Mann and NBody
distributions do not use this strategy and instead a class is created for each
which captures this functionality
"""

cpdef double adist_correlated(double mu, double Eout, double[::1] Eout_points,
                              list adists):
    cdef int i, last_Eout
    cdef double f

    last_Eout = len(Eout_points) - 1
    # Find the outgoing energy bin of interest
    if Eout >= Eout_points[last_Eout]:
        i = len(adists) - 2
    elif Eout <= Eout_points[0]:
        i = 0
    else:
        i = bisect(Eout_points, Eout) - 1

    # Pick the nearest angular distribution (consistent with OpenMC)
    # Make sure the Eout points are not the same value (this happens in a
    # few evaluations). If they are the same, just use the lower
    # distribution.
    f = (Eout_points[i + 1] - Eout_points[i])
    if f > 0.:
        # Now that we made sure the consective Eout points are not the same
        # continue finding the interpolant to see which distribution we
        # are closer to
        f = (Eout - Eout_points[i]) / f
        if f > 0.5:
            i += 1

    # Now evaluate and return our resultant angular distribution
    adist = adists[i]
    return adist._eval(mu)


cpdef double adist_isotropic(double mu, double Eout):
    if mu >= -1. and mu <= 1.:
        return 0.5
    else:
        return 0.


cpdef double unity(double x):
    return 1.