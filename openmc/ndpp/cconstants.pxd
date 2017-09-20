cimport numpy as np

cdef double _PI

# The quadrature points can be gathered now, we will get them between 0 and 1
# so the shifting to the proper Elo, Ehi bounds is easier later
cdef int _N_QUAD
cdef double[::1] _POINTS
cdef double[::1] _WEIGHTS
cdef double[::1] _FMU
cdef double[::1] _MU

# The number of outgoing energy points to use when integrating the free-gas
# kernel; we are using simpson's 3/8 rule so this needs to be a multiple of 3
cdef int _N_EOUT
cdef double _N_EOUT_DOUBLE

# Set our simpson 3/8 rule coefficients
cdef double[::1] _SIMPSON_WEIGHTS
cdef int index

# Angular distribution types
cdef int _ADIST_TYPE_TABULAR, _ADIST_TYPE_UNIFORM, _ADIST_TYPE_DISCRETE

# CONSTANTS USED IN PLACE OF openmc.stats.Univariate interpolation types
cdef int HISTOGRAM
cdef int LINLIN
cdef int LINLOG
cdef int LOGLIN
cdef int LOGLOG
