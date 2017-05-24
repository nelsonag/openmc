#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

from libc.math cimport exp, log
from collections import Iterable
import numpy as np
cimport numpy as np

from openmc.stats.bisect cimport bisect


def tabulated1d_call(this, x):
    # Handle both array and scalar input
    if isinstance(x, Iterable):
        return np.fromiter((tabulated1d_eval(this, x_i) for x_i in x),
                           np.float, len(x))
    else:
        return tabulated1d_eval(this, x)


cdef double tabulated1d_eval(object this, double x):
    cdef double y, xk, xi, xi1, yi, yi1
    cdef size_t idx, end_x
    cdef long k

    end_x = len(this._x) - 1

    if this._x[0] - 1e-14 <= x <= this._x[0] + 1e-14:
        return this._y[0]
    elif this._x[end_x] - 1e-14 <= x <= this._x[end_x] + 1e-14:
        return this._y[end_x]

    # Get indices for interpolation
    if x <= this._x[0]:
        idx = 0
    elif x >= this._x[end_x]:
        idx = len(this._x) - 2
    else:
        idx = bisect(this._x, x) - 1

    if this._breakpoints.shape[0] == 1:
        k = 0
    else:
        k = bisect(this._breakpoints, idx) - 1

    xk = x
    xi = this._x[idx]
    xi1 = this._x[idx + 1]
    yi = this._y[idx]
    yi1 = this._y[idx + 1]

    if this._interpolation[k] == 1:
        # Histogram
        y = yi

    elif this._interpolation[k] == 2:
        # Linear-linear
        y = yi + (xk - xi) / (xi1 - xi) * (yi1 - yi)

    elif this._interpolation[k] == 3:
        # Linear-log
        y = yi + log(xk / xi) / log(xi1 / xi) * (yi1 - yi)

    elif this._interpolation[k] == 4:
        # Log-linear
        y = yi * exp((xk - xi) / (xi1 - xi) * log(yi1 / yi))

    elif this._interpolation[k] == 5:
        # Log-log
        y = (yi * exp(log(xk / xi) / log(xi1 / xi) * log(yi1 / yi)))

    return y


cpdef double tabulated1d_integrate(object this, double lo, double hi):
    cdef size_t i, i_start, i_end, k, end_x
    cdef double result, xi, xi1, xlo, xhi, plo, phi, pi, pi1, m
    end_x = len(this._x) - 1
    result = 0.

    # Find the bounds of integration we care about
    if lo <= this._x[0]:
        i_start = 0
    elif lo >= this._x[end_x]:
        i_start = len(this._x) - 2
    else:
        i_start = bisect(this._x, lo) - 1

    if hi <= this._x[0]:
        i_end = 0
    elif hi >= this._x[end_x]:
        i_end = len(this._x) - 2
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

        # Find our interpolation region
        if this._breakpoints.shape[0] == 1:
            k = 0
        else:
            k = bisect(this._breakpoints, i) - 1

        if this._interpolation[k] == 1:
            # Histogram integration
            result += (xhi - xlo) * this._p[i]
        elif this._interpolation[k] == 2:
            # Linear-linear interpolation
            # Get our end points first (we could just use __call__, but
            # then we'd execute lots of the same code all over again)
            plo = pi + (xlo - xi) / (xi1 - xi) * (pi1 - pi)
            phi = pi + (xhi - xi) / (xi1 - xi) * (pi1 - pi)
            result += 0.5 * (xhi - xlo) * (plo + phi)

        elif this._interpolation[k] == 3:
            # Linear-log interpolation
            # Get our end points first
            plo = pi + log(xlo / xi) / log(xi1 / xi) * (pi1 - pi)
            phi = pi + log(xhi / xi) / log(xi1 / xi) * (pi1 - pi)
            m = (pi1 - pi) / log(xi1 / xi)
            result += m * xhi * log(xhi / xlo) + (xhi - xlo) * (plo - m)

        elif this._interpolation[k] == 4:
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

        elif this._interpolation[k] == 5:
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
