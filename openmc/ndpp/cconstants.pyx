from scipy.special import roots_sh_legendre
import numpy as np

_PI = np.pi

# The quadrature points can be gathered now, we will get them between 0 and 1
# so the shifting to the proper Elo, Ehi bounds is easier later
_N_QUAD = 20
_POINTS, _WEIGHTS = roots_sh_legendre(_N_QUAD)
_FMU = np.empty(_N_QUAD)
_MU = np.empty(_N_QUAD)

# The number of outgoing energy points to use when integrating the free-gas
# kernel; we are using simpson's 3/8 rule so this needs to be a multiple of 3
_N_EOUT = 201
_N_EOUT_DOUBLE = <double> _N_EOUT

# Set our simpson 3/8 rule coefficients
_SIMPSON_WEIGHTS = np.empty(_N_EOUT)
for index in range(_N_EOUT):
    if (index == 0) or (index == _N_EOUT - 1):
        _SIMPSON_WEIGHTS[index] = 0.375
    elif index % 3 == 0:
        _SIMPSON_WEIGHTS[index] = 0.375 * 2.
    else:
        _SIMPSON_WEIGHTS[index] = 0.375 * 3.

# Minimum value of c, the constant used when converting inelastic CM
# distributions to the lab-frame
_MIN_C = 25.
_MIN_C2 = _MIN_C * _MIN_C

# Angular distribution types
_ADIST_TYPE_TABULAR = 0
_ADIST_TYPE_UNIFORM = 1
_ADIST_TYPE_DISCRETE = 2

# CONSTANTS USED IN PLACE OF openmc.stats.Univariate interpolation types
HISTOGRAM = 1
LINLIN = 2
LINLOG = 3
LOGLIN = 4
LOGLOG = 5
