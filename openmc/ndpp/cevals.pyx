#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: fast_gil=True

cdef double tabular_eval(double[:] this_x, double[:] this_p,
                                int interpolation, double x):
    cdef size_t i, end_x
    cdef double xi, xi1, pi, pi1, y

    end_x = this_x.shape[0] - 1

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

    if interpolation == HISTOGRAM:
        y = pi
    elif interpolation == LINLIN:
        y = pi + (x - xi) / (xi1 - xi) * (pi1 - pi)
    elif interpolation == LINLOG:
        y = pi + log(x / xi) / log(xi1 / xi) * (pi1 - pi)
    elif interpolation == LOGLIN:
        y = pi * exp((x - xi) / (xi1 - xi) * log(pi1 / pi))
    elif interpolation == LOGLOG:
        y = pi * exp(log(x / xi) / log(xi1 / xi) * log(pi1 / pi))

    return y


cdef double tabulated1d_eval(double[:] this_x, double[:] this_y,
                                    long[:] breakpoints, long[:] interpolation,
                                    double x):
    cdef double y, xk, xi, xi1, yi, yi1
    cdef size_t idx, end_x, k

    end_x = this_x.shape[0] - 1

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

    if interpolation[k] == HISTOGRAM:
        y = yi

    elif interpolation[k] == LINLIN:
        y = yi + (xk - xi) / (xi1 - xi) * (yi1 - yi)

    elif interpolation[k] == LINLOG:
        y = yi + log(xk / xi) / log(xi1 / xi) * (yi1 - yi)

    elif interpolation[k] == LOGLIN:
        y = yi * exp((xk - xi) / (xi1 - xi) * log(yi1 / yi))

    elif interpolation[k] == LOGLOG:
        y = (yi * exp(log(xk / xi) / log(xi1 / xi) * log(yi1 / yi)))

    return y


cdef double tabulated1d_eval_w_search_params(double[:] this_x, double[:] this_y,
                                             long[:] breakpoints,
                                             long[:] interpolation, double x,
                                             size_t idx, int interp_type):
    cdef double y, xk, xi, xi1, yi, yi1
    cdef size_t end_x, k

    end_x = this_x.shape[0] - 1

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
    interp_type = interpolation[k]

    xk = x
    xi = this_x[idx]
    xi1 = this_x[idx + 1]
    yi = this_y[idx]
    yi1 = this_y[idx + 1]

    if interp_type == HISTOGRAM:
        y = yi

    elif interp_type == LINLIN:
        y = yi + (xk - xi) / (xi1 - xi) * (yi1 - yi)

    elif interp_type == LINLOG:
        y = yi + log(xk / xi) / log(xi1 / xi) * (yi1 - yi)

    elif interp_type == LOGLIN:
        y = yi * exp((xk - xi) / (xi1 - xi) * log(yi1 / yi))

    elif interp_type == LOGLOG:
        y = (yi * exp(log(xk / xi) / log(xi1 / xi) * log(yi1 / yi)))

    return y


cdef double fast_tabulated1d_eval(double x, double xi, double xi1, double yi,
                                  double yi1, int interp_type):
    cdef double y
    if interp_type == HISTOGRAM:
        y = yi
    elif interp_type == LINLIN:
        y = yi + (x - xi) / (xi1 - xi) * (yi1 - yi)
    elif interp_type == LINLOG:
        y = yi + log(x / xi) / log(xi1 / xi) * (yi1 - yi)
    elif interp_type == LOGLIN:
        y = yi * exp((x - xi) / (xi1 - xi) * log(yi1 / yi))
    elif interp_type == LOGLOG:
        y = (yi * exp(log(x / xi) / log(xi1 / xi) * log(yi1 / yi)))
    return y
