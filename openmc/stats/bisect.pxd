cimport cython

cdef size_t bisect(double[:] a, double x, size_t lo=*, size_t hi=*)
cdef size_t bisect_int(cython.integral[:] a, cython.integral x, size_t lo=*,
                       size_t hi=*)
