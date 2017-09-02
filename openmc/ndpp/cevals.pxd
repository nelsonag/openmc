#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False

from libc.math cimport exp, log

from openmc.stats.bisect cimport bisect, bisect_int

from .cconstants cimport HISTOGRAM, LINLIN, LINLOG, LOGLIN, LOGLOG

cdef double discrete_eval(double[:] this_x, double[:] this_p, size_t end_x,
                          double x)


cdef double tabular_eval(double[:] this_x, double[:] this_p, size_t end_x,
                         int interpolation, double x)


cdef double tabular_eval_w_search_params(double[:] this_x, double[:] this_p,
                                         size_t end_x, int interpolation,
                                         double x, size_t* i)


cdef double tabulated1d_eval(double[:] this_x, double[:] this_y, size_t end_x,
                                    long[:] breakpoints, long[:] interpolation,
                                    double x)

cdef double tabulated1d_eval_w_search_params(double[:] this_x, double[:] this_y,
                                             size_t end_x, long[:] breakpoints,
                                             long[:] interpolation, double x,
                                             size_t* idx, int* interp_type)

cdef double fast_tabulated1d_eval(double x, double xi, double xi1, double yi,
                                  double yi1, int interp_type)
