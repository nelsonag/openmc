#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False


cdef double discrete_eval(double[:] this_x, double[:] this_p, size_t end_x,
                          double x):
    cdef size_t i

    for i in range(end_x + 1):
        if x == this_x[i]:
            return this_p[i]
    return 0.


cdef double tabular_eval(double[:] this_x, double[:] this_p, size_t end_x,
                         int interpolation, double x):
    cdef size_t i
    cdef double xi, xi1, pi, pi1, y

    if x < this_x[0] or x > this_x[end_x]:
        return 0.

    if this_x[0] - 1e-14 <= x <= this_x[0] + 1e-14:
        return this_p[0]
    elif this_x[end_x] - 1e-14 <= x <= this_x[end_x] + 1e-14:
        return this_p[end_x]

    # Get index for interpolation and interpolant
    # if x == this_x[0]:
    #     return this_p[0]
    # elif x == this_x[end_x]:
    #     return this_p[end_x]
    # else:
    #     i = bisect(this_x, x, lo=0, hi=end_x + 1) - 1
    i = bisect(this_x, x, lo=0, hi=end_x + 1) - 1
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
    else:
        y = pi * exp(log(x / xi) / log(xi1 / xi) * log(pi1 / pi))

    return y


cdef double tabular_eval_w_search_params(double[:] this_x, double[:] this_p,
                                         size_t end_x, int interpolation,
                                         double x, size_t* i):
    cdef double xi, xi1, pi, pi1, y

    if x < this_x[0] or x > this_x[end_x]:
        i[0] = 0
        return 0.

    if this_x[0] - 1e-14 <= x <= this_x[0] + 1e-14:
        i[0] = 0
        return this_p[0]
    elif this_x[end_x] - 1e-14 <= x <= this_x[end_x] + 1e-14:
        i[0] = end_x
        return this_p[end_x]

    # Get index for interpolation and interpolant
    # if x == this_x[0]:
    #     i[0] = 0
    #     return this_p[0]
    # elif x == this_x[end_x]:
    #     i[0] = end_x
    #     return this_p[end_x]
    # else:
    #     i[0] = bisect(this_x, x, lo=0, hi=end_x + 1) - 1
    i[0] = bisect(this_x, x, lo=0, hi=end_x + 1) - 1
    xi = this_x[i[0]]
    xi1 = this_x[i[0] + 1]
    pi = this_p[i[0]]
    pi1 = this_p[i[0] + 1]

    if interpolation == HISTOGRAM:
        y = pi
    elif interpolation == LINLIN:
        y = pi + (x - xi) / (xi1 - xi) * (pi1 - pi)
    elif interpolation == LINLOG:
        y = pi + log(x / xi) / log(xi1 / xi) * (pi1 - pi)
    elif interpolation == LOGLIN:
        y = pi * exp((x - xi) / (xi1 - xi) * log(pi1 / pi))
    else:
        y = pi * exp(log(x / xi) / log(xi1 / xi) * log(pi1 / pi))

    return y


cdef double tabulated1d_eval(double[:] this_x, double[:] this_y, size_t end_x,
                             long[:] breakpoints, long[:] interpolation,
                             double x):
    cdef double y, xk, xi, xi1, yi, yi1
    cdef size_t idx, k

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
        idx = bisect(this_x, x, lo=0, hi=end_x + 1) - 1

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

    else:
        y = (yi * exp(log(xk / xi) / log(xi1 / xi) * log(yi1 / yi)))

    return y


cdef double tabulated1d_eval_w_search_params(double[:] this_x, double[:] this_y,
                                             size_t end_x, long[:] breakpoints,
                                             long[:] interpolation, double x,
                                             size_t* idx, int* interp_type):
    cdef double y, xk, xi, xi1, yi, yi1
    cdef size_t k

    if this_x[0] - 1e-14 <= x <= this_x[0] + 1e-14:
        idx[0] = 0
        interp_type[0] = interpolation[0]
        return this_y[0]
    elif this_x[end_x] - 1e-14 <= x <= this_x[end_x] + 1e-14:
        idx[0] = end_x
        interp_type[0] = interpolation[breakpoints.shape[0] - 1]
        return this_y[end_x]

    # Get indices for interpolation
    if x <= this_x[0]:
        idx[0] = 0
    elif x >= this_x[end_x]:
        idx[0] = end_x - 1
    else:
        idx[0] = bisect(this_x, x, lo=0, hi=end_x + 1) - 1

    if breakpoints.shape[0] == 1:
        k = 0
    else:
        k = bisect_int(breakpoints, idx[0]) - 1
    interp_type[0] = interpolation[k]

    xk = x
    xi = this_x[idx[0]]
    xi1 = this_x[idx[0] + 1]
    yi = this_y[idx[0]]
    yi1 = this_y[idx[0] + 1]

    if interp_type[0] == HISTOGRAM:
        y = yi

    elif interp_type[0] == LINLIN:
        y = yi + (xk - xi) / (xi1 - xi) * (yi1 - yi)

    elif interp_type[0] == LINLOG:
        y = yi + log(xk / xi) / log(xi1 / xi) * (yi1 - yi)

    elif interp_type[0] == LOGLIN:
        y = yi * exp((xk - xi) / (xi1 - xi) * log(yi1 / yi))

    else:
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
    else:
        y = (yi * exp(log(x / xi) / log(xi1 / xi) * log(yi1 / yi)))
    return y
