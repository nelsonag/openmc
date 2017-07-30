#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False


cdef size_t bisect(double[:] a, double x, size_t lo=0, size_t hi=0):
    cdef size_t mid
    if hi == 0:
        hi = len(a)

    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


cdef size_t bisect_int(cython.integral[:] a, cython.integral x, size_t lo=0,
                       size_t hi=0):
    cdef size_t mid
    if hi == 0:
        hi = len(a)

    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo
