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
    cdef int i
    for i in range(this._x.shape[0]):
        if x == this._x[i]:
            return this._p[i]
    return 0.

cpdef double discrete_integrate(object this, double lo, double hi):
    cdef size_t i
    cdef double result
    result = 0.
    for i in range(this._x.shape[0]):
        if lo <= this._x[i] <= hi:
            result += this._p[i]
    return result


cpdef double uniform_call(object this, double x):
    cdef int i
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
    cdef double xi, xi1, pi, pi1, y

    end_x = len(this._x) - 1

    if x < this._x[0] or x > this._x[end_x]:
        return 0.

    if this._x[0] - 1e-14 <= x <= this._x[0] + 1e-14:
        return this._p[0]
    elif this._x[end_x] - 1e-14 <= x <= this._x[end_x] + 1e-14:
        return this._p[end_x]

    # Get index for interpolation and interpolant
    # Get indices for interpolation
    if x == this._x[0]:
        return this._p[0]
    elif x == this._x[end_x]:
        return this._p[end_x]
    else:
        i = bisect(this._x, x) - 1
    xi = this._x[i]
    xi1 = this._x[i + 1]
    pi = this._p[i]
    pi1 = this._p[i + 1]

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
    cdef size_t i, i_start, i_end, num_x
    cdef double result, xi, xi1, xlo, xhi, plo, phi, pi, pi1, m
    result = 0.
    end_x = len(this._x) - 1

    # Find the bounds of integration we care about
    if lo <= this._x[0]:
        lo = this._x[0]
        i_start = 0
    elif lo >= this._x[end_x]:
        return result
    else:
        i_start = bisect(this._x, lo) - 1

    if hi <= this._x[0]:
        return result
    elif hi >= this._x[end_x]:
        hi = this._x[end_x]
        i_end = end_x - 1
    else:
        i_end = bisect(this._x, hi) - 1

    for i in range(i_start, i_end + 1):
        xi = this._x[i]
        xi1 = this._x[i + 1]
        if i == i_start:
            xlo = lo
        else:
            xlo = xi
        if i == i_end:
            xhi = hi
        else:
            xhi = xi1

        pi = this._p[i]
        pi1 = this._p[i + 1]

        if this._interpolation == 'histogram':
            # Histogram integration
            result += (xhi - xlo) * this._p[i]
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
