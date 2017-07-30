#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

from libc.math cimport exp, log
from collections import Iterable
import numpy as np
cimport numpy as np

from openmc.stats.bisect cimport bisect

cdef double tabulated1d_eval(object this, double x)

cdef tabulated1d_vector_eval(object this,
                             np.ndarray[np.float64_t, ndim=1] x,
                             size_t vec_size)

cpdef double tabulated1d_integrate(object this, double lo, double hi)