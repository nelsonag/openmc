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

cpdef double tabulated1d_integrate(object this, double lo, double hi)