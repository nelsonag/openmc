#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

from libc.math cimport exp, log
from collections import Iterable
import numpy as np
cimport numpy as np

from .bisect cimport bisect


cpdef double discrete_call(object this, double x):
    cdef size_t i
    cdef double[:] this_x = this._x

    for i in range(this_x.shape[0]):
        if x == this_x[i]:
            return this._p[i]
    return 0.

cpdef double discrete_integrate(object this, double lo, double hi):
    cdef size_t i
    cdef double[:] this_x = this._x
    cdef double result
    result = 0.
    for i in range(this_x.shape[0]):
        if lo <= this_x[i] <= hi:
            result += this._p[i]
    return result


cpdef double uniform_call(object this, double x):
    cdef size_t i
    if this._a <= x <= this._b:
        return 1. / (this._b - this._a)
    else:
        return 0.


cpdef double uniform_integrate(object this, double lo, double hi):
    cdef size_t i
    cdef double result, low_bound, high_bound
    low_bound = max(this._a, lo)
    high_bound = min(this._b, hi)
    return (high_bound - low_bound) / (this._b - this._a)


cpdef double tabular_call(object this, double x):
    cdef size_t i, end_x
    cdef double[:] this_x = this._x
    cdef double[:] this_p = this._p
    cdef double xi, xi1, pi, pi1, y

    end_x = len(this_x) - 1

    if x < this_x[0] or x > this_x[end_x]:
        return 0.

    if this_x[0] - 1e-14 <= x <= this_x[0] + 1e-14:
        return this_p[0]
    elif this_x[end_x] - 1e-14 <= x <= this_x[end_x] + 1e-14:
        return this_p[end_x]

    # Get index for interpolation and interpolant
    # Get indices for interpolation
    if x == this_x[0]:
        return this_p[0]
    elif x == this_x[end_x]:
        return this_p[end_x]
    else:
        i = bisect(this_x, x) - 1

    xi = this_x[i]
    xi1 = this_x[i + 1]
    pi = this_p[i]
    pi1 = this_p[i + 1]

    if this._interpolation == 'histogram':
        y = pi
    elif this._interpolation == 'linear-linear':
        y = pi + (x - xi) / (xi1 - xi) * (pi1 - pi)
    elif this._interpolation == 'linear-log':
        y = pi + log(x / xi) / log(xi1 / xi) * (pi1 - pi)
    elif this._interpolation == 'log-linear':
        y = pi * exp((x - xi) / (xi1 - xi) * log(pi1 / pi))
    elif this._interpolation == 'log-log':
        y = pi * exp(log(x / xi) / log(xi1 / xi) * log(pi1 / pi))

    return y


cpdef double tabular_integrate(object this, double lo, double hi):
    cdef size_t i, i_start, i_end, end_x
    cdef double[:] this_x = this._x
    cdef double[:] this_p = this._p
    cdef double result, xi, xi1, xlo, xhi, plo, phi, pi, pi1, m

    result = 0.
    end_x = len(this_x) - 1

    # Find the bounds of integration we care about
    if lo <= this_x[0]:
        lo = this_x[0]
        i_start = 0
    elif lo >= this_x[end_x]:
        return result
    else:
        i_start = bisect(this_x, lo) - 1

    if hi <= this_x[0]:
        return result
    elif hi >= this_x[end_x]:
        hi = this_x[end_x]
        i_end = end_x - 1
    else:
        i_end = bisect(this_x, hi) - 1

    for i in range(i_start, i_end + 1):
        xi = this_x[i]
        xi1 = this_x[i + 1]
        if i == i_start:
            xlo = lo
        else:
            xlo = xi
        if i == i_end:
            xhi = hi
        else:
            xhi = xi1

        pi = this_p[i]
        pi1 = this_p[i + 1]

        if this._interpolation == 'histogram':
            # Histogram integration
            result += (xhi - xlo) * this_p[i]
        elif this._interpolation == 'linear-linear':
            # Linear-linear interpolation
            # Get our end points first (we could just use __call__, but
            # then we'd execute lots of the same code all over again)
            plo = pi + (xlo - xi) / (xi1 - xi) * (pi1 - pi)
            phi = pi + (xhi - xi) / (xi1 - xi) * (pi1 - pi)
            result += 0.5 * (xhi - xlo) * (plo + phi)

        elif this._interpolation == 'linear-log':
            # Linear-log interpolation
            # Get our end points first
            plo = pi + log(xlo / xi) / log(xi1 / xi) * (pi1 - pi)
            phi = pi + log(xhi / xi) / log(xi1 / xi) * (pi1 - pi)
            m = (pi1 - pi) / log(xi1 / xi)
            result += m * xhi * log(xhi / xlo) + (xhi - xlo) * (plo - m)

        elif this._interpolation == 'log-linear':
            # Log-linear interpolation
            # Get our end points first
            plo = pi * exp((xlo - xi) / (xi1 - xi) * log(pi1 / pi))
            phi = pi * exp((xhi - xi) / (xi1 - xi) * log(pi1 / pi))
            m = log(pi1 / pi) / (xi1 - xi)
            if m == 0.:
                result += plo * (xhi - xlo)
            else:
                result += (plo * exp(-m * xlo)) * \
                    (exp(m * xhi) - exp(m * xlo)) / m

        elif this._interpolation == 'log-log':
            # Log-log interpolation
            # Get our end points first
            plo = pi * exp(log(plo / xi) / log(xi1 / xi) * log(pi1 / pi))
            phi = pi * exp(log(phi / xi) / log(xi1 / xi) * log(pi1 / pi))
            m = log(pi1 / pi) / log(xi1 / xi)
            if m == -1.:
                result += plo * xlo * log(xhi / xlo)
            else:
                result += plo / (m + 1.) * (xhi * (xhi / xlo)**m - xlo)

    return result
