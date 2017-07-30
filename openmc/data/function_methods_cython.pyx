#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

from libc.math cimport exp, log
from collections import Iterable
import numpy as np
cimport numpy as np
from cython.view cimport array as cvarray

from openmc.stats.bisect cimport bisect, bisect_int


def tabulated1d_call(this, x):
    # Handle both array and scalar input
    if isinstance(x, Iterable):
        return tabulated1d_vector_eval(this, np.asarray(x), len(x))
    else:
        return tabulated1d_eval(this, x)


cdef double tabulated1d_eval(object this, double x):
    cdef double y, xk, xi, xi1, yi, yi1
    cdef size_t idx, end_x
    cdef int k
    cdef double[:] this_x = this._x
    cdef double[:] this_y = this._y
    cdef int[:] breakpoints = this._breakpoints
    cdef int[:] interpolation = this._interpolation

    end_x = len(this_x) - 1

    if this_x[0] - 1e-14 <= x <= this_x[0] + 1e-14:
        return this_y[0]
    elif this_x[end_x] - 1e-14 <= x <= this_x[end_x] + 1e-14:
        return this_y[end_x]

    # Get indices for interpolation
    if x <= this_x[0]:
        idx = 0
    elif x >= this_x[end_x]:
        idx = end_x - 1
    else:
        idx = bisect(this_x, x) - 1

    if breakpoints.shape[0] == 1:
        k = 0
    else:
        k = bisect_int(breakpoints, idx) - 1

    xk = x
    xi = this_x[idx]
    xi1 = this_x[idx + 1]
    yi = this_y[idx]
    yi1 = this_y[idx + 1]

    if interpolation[k] == 1:
        # Histogram
        y = yi

    elif interpolation[k] == 2:
        # Linear-linear
        y = yi + (xk - xi) / (xi1 - xi) * (yi1 - yi)

    elif interpolation[k] == 3:
        # Linear-log
        y = yi + log(xk / xi) / log(xi1 / xi) * (yi1 - yi)

    elif interpolation[k] == 4:
        # Log-linear
        y = yi * exp((xk - xi) / (xi1 - xi) * log(yi1 / yi))

    elif interpolation[k] == 5:
        # Log-log
        y = (yi * exp(log(xk / xi) / log(xi1 / xi) * log(yi1 / yi)))

    return y


cdef tabulated1d_vector_eval(object this,
                             np.ndarray[np.float64_t, ndim=1] x, size_t vec_size):
    cdef double[:] y = cvarray(shape=(vec_size,), itemsize=sizeof(double),
                               format="d")
    cdef double xi, xi1, yi, yi1
    cdef size_t idx, end_x
    cdef int k, j
    cdef double[:] this_x = this._x
    cdef double[:] this_y = this._y
    cdef int[:] breakpoints = this._breakpoints
    cdef int[:] interpolation = this._interpolation

    end_x = len(this_x) - 1

    for j in range(vec_size):
        if this_x[0] - 1e-14 <= x[j] <= this_x[0] + 1e-14:
            y[j] = this_y[0]
            continue
        elif this_x[end_x] - 1e-14 <= x[j] <= this_x[end_x] + 1e-14:
            y[j] = this_y[end_x]
            continue

        # Get indices for interpolation
        if x[j]<= this_x[0]:
            idx = 0
        elif x[j] >= this_x[end_x]:
            idx = end_x - 1
        else:
            idx = bisect(this_x, x[j]) - 1

        if breakpoints.shape[0] == 1:
            k = 0
        else:
            k = bisect_int(breakpoints, idx) - 1

        xi = this_x[idx]
        xi1 = this_x[idx + 1]
        yi = this_y[idx]
        yi1 = this_y[idx + 1]

        if interpolation[k] == 1:
            # Histogram
            y[j] = yi

        elif interpolation[k] == 2:
            # Linear-linear
            y[j] = yi + (x[j] - xi) / (xi1 - xi) * (yi1 - yi)

        elif interpolation[k] == 3:
            # Linear-log
            y[j] = yi + log(x[j] / xi) / log(xi1 / xi) * (yi1 - yi)

        elif interpolation[k] == 4:
            # Log-linear
            y[j] = yi * exp((x[j] - xi) / (xi1 - xi) * log(yi1 / yi))

        elif interpolation[k] == 5:
            # Log-log
            y[j] = (yi * exp(log(x[j] / xi) / log(xi1 / xi) * log(yi1 / yi)))

    return np.asarray(y)


cpdef double tabulated1d_integrate(object this, double lo, double hi):
    cdef size_t i, i_start, i_end, k, end_x
    cdef double result, xi, xi1, xlo, xhi, plo, phi, pi, pi1, m
    cdef double[:] this_x = this._x
    cdef double[:] this_p = this._p
    cdef int[:] breakpoints = this._breakpoints
    cdef int[:] interpolation = this._interpolation

    end_x = len(this_x) - 1
    result = 0.

    # Find the bounds of integration we care about
    if lo <= this_x[0]:
        i_start = 0
    elif lo >= this_x[end_x]:
        i_start = end_x - 1
    else:
        i_start = bisect(this_x, lo) - 1

    if hi <= this_x[0]:
        i_end = 0
    elif hi >= this_x[end_x]:
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

        # Find our interpolation region
        if breakpoints.shape[0] == 1:
            k = 0
        else:
            k = bisect_int(breakpoints, i) - 1

        if interpolation[k] == 1:
            # Histogram integration
            result += (xhi - xlo) * this_p[i]
        elif interpolation[k] == 2:
            # Linear-linear interpolation
            # Get our end points first (we could just use __call__, but
            # then we'd execute lots of the same code all over again)
            plo = pi + (xlo - xi) / (xi1 - xi) * (pi1 - pi)
            phi = pi + (xhi - xi) / (xi1 - xi) * (pi1 - pi)
            result += 0.5 * (xhi - xlo) * (plo + phi)

        elif interpolation[k] == 3:
            # Linear-log interpolation
            # Get our end points first
            plo = pi + log(xlo / xi) / log(xi1 / xi) * (pi1 - pi)
            phi = pi + log(xhi / xi) / log(xi1 / xi) * (pi1 - pi)
            m = (pi1 - pi) / log(xi1 / xi)
            result += m * xhi * log(xhi / xlo) + (xhi - xlo) * (plo - m)

        elif interpolation[k] == 4:
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

        elif interpolation[k] == 5:
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
