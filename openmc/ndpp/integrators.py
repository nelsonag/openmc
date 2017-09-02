import numpy as np
import scipy.optimize as sopt

import openmc.stats
from openmc.data import KalbachMann, UncorrelatedAngleEnergy, \
    CorrelatedAngleEnergy, LevelInelastic, NBodyPhaseSpace, Uniform
from .interpolators import *
from .cython_integrators import *
from .uncorrelated import Uncorrelated
from .correlated import Correlated
from .kalbach import KM
from .nbody import NBody
from .freegas import *


# The free-gas kernel (FGK) functions asymptotically go to zero as outgoing
# energy varies for a given value of mu. We dont want to integrate over all
# that space. The following constants are used in identifying a more useful
# range to integrate over

# This term sets the fraction of the peak (for a given mu) of the FGK function
# to look for when finding the lower and upper Eout values of integration
_FGK_ROOT_TOL = 1.e-2

# These are constants we initialize once instead of every time through that are
# used during the search for the FGK roots
# The mu value to use on the back-scatter end to find the roots
_MU_BACK = np.array([-1.])
# The mu value to use on the front-scatter end to find the roots
# We dont use mu = 1. since the function behaves poorly there
_MU_FWD = np.array([0.9])
# Set the mu interpolant to extend the range found at _MU_BACK and _MU_BACK
# all the way to a forward-scatter mu of 1.0
_INTERP_MU = (_MU_FWD[0] - (_MU_BACK[0])) * 0.5
# Store space for our results
_FGK_RESULT = np.empty_like(_MU_BACK)


def integrate(Ein, this, Eouts, scatter_format, cm, awr, freegas_cutoff, kT,
              q_value, mus=None, order=None, xs=None, mus_grid=None,
              wgts=None):
    """Integrates this distribution at a given incoming energy,
    over a given lab-frame mu range and given outgoing energy bounds.

    Parameters
    ----------
    Ein : float
        Incoming energy, in units of eV
    this : openmc.AngleEnergy
        Distribution to be integrated
    Eouts : Iterable of float
        Outgoing energy bins (in units of eV) to integrate across
    scatter_format : {'legendre' or 'histogram'}
        Mathematical scatter_format of the scattering matrices: either a
        Legendre expansion of the order defined in the order parameter, or
        a histogram scatter_format with a number of bins defined by the order
        parameter.
    cm : bool
        Flag stating whether or not the reaction providing the distribution
        is in the center of mass frame (True) or not.  Defaults to False.
    awr : float
        Atomic weight ratio; optional. Required if the distribution is
        provided in the center-of-mass frame.
    freegas_cutoff : float
        Maximal energy (in eV) in which the free-gas kernel is applied;
        defaults to 400 eV for most nuclides, but 20 MeV for H-1.
    kT : float
        Temperature (in units of eV) of the material; defaults to 2.53E-2 eV.
    q_value : float
        The Q-value of this reaction in eV.
    mus : Iterable of float; optional
        Lab-frame angular bins to integrate across. Defaults to [-1., 1.]
        which integrates across the entire mu domain. Required if
        scatter_format is 'histogram.'
    order : int, optional
        Order of the Legendre expansion of the result; the value of this
        parameter + 1 sets the dimensionality of the 3rd index of the
        result. Required if scatter_format is 'histogram.'
    xs : openmc.data.Tabulated1D, optional
        Cross sections. Only used if performing free-gas kernel integration.
    mus_grid : numpy.ndarray of float
        Array of lab-frame mu values to use when quadrature integrating the
        freegas kernel; passed in here to avoid re-calculating at every Ein;
        only required for Legendre scattering
    wgts : numpy.ndarray of float
        wgts corresponding to the mus_grid; also passed in to avoid
        re-calculating at every Ein; only required for Legendre scattering

    Returns
    -------
    np.ndarray
        Resultant integral with the following dimensions:
        [# energy bins, # mu bins or expansion orders]

    """

    # First deal with the default value of mus (None), if applicable
    if scatter_format == 'histogram':
        order = None
        if mus is None:
            mus = np.array([-1., 1.])

        # Set mus_grid and wgts to a value so they wont fail as not None in
        # the cython
        mus_grid = np.array([0.])
        wgts = np.array([[0.]])

    elif scatter_format == 'legendre' and order is None:
        # Then make sure the order is similarly consistent
        raise ValueError("order cannot be 'None' if scatter_format is "
                         "'legendre'")

    # Check first for two-body kinematics
    if isinstance(this, UncorrelatedAngleEnergy):
        if this.energy is None or isinstance(this.energy, LevelInelastic):
            if Ein <= freegas_cutoff and q_value == 0.:
                return _integrate_twobody_cm_freegas(this, Ein, Eouts, order,
                                                     mus, awr, kT, xs,
                                                     mus_grid, wgts)
            else:
                return _integrate_twobody_cm_TAR(this, Ein, Eouts, awr,
                                                 q_value, order, mus)

    # Now move on to general center-of-mass or lab-frame integration
    if cm:
        return _integrate_generic_cm(this, Ein, Eouts, awr, order, mus,
                                     mus_grid, wgts)
    else:
        return _integrate_generic_lab(this, Ein, Eouts, order, mus, mus_grid,
                                      wgts)


###############################################################################
# DATA PREPARATON
###############################################################################


def _preprocess_kalbach(this, Ein):
    """Performs the interpolation and builds the energy and angular
    distributions for use in the integration routines
    """

    # Generate the distribution via interpolation
    edist, precompound, slope = interpolate_kalbach(this, Ein)

    return KM(slope, precompound, edist)


def _preprocess_uncorr(this, Ein):
    """Performs the interpolation and builds the energy and angular
    distributions for use in the integration routines
    """

    # Do not proceed if we have a Discrete angular distribution;
    # This will require a special integration routine which until encountered,
    # has not been implemented.
    # This check assumes all angular distributuons within this._mu will be of
    # the same type

    if this._angle and isinstance(this._angle._mu[0], openmc.stats.Discrete):
        raise NotImplementedError("Integration of a discrete angular "
                                  "distribution in the center-of-mass "
                                  "has not been implemented")

    # Generate the distribution via interpolation
    edist, adist = interpolate_uncorr(this, Ein)

    return Uncorrelated(adist, edist)


def _preprocess_corr(this, Ein):
    """Performs the interpolation and builds the energy and angular
    distributions for use in the integration routines
    """

    # Do not proceed if we have a Discrete angular distribution;
    # This will require a special integration routine which until encountered,
    # has not been implemented.
    # This check assumes all angular distributuons within this._mu will be of
    # the same type
    if isinstance(this._mu[0], openmc.stats.Discrete):
        raise NotImplementedError("Integration of a discrete angular "
                                  "distribution has not been implemented")

    # Generate the distribution via interpolation
    edist, adists = interpolate_corr(this, Ein)

    return Correlated(adists, edist)


def _preprocess_nbody(this, Ein):
    """Performs the interpolation and builds the energy and angular
    distributions for use in the integration routines
    """

    return NBody(this, Ein)


###############################################################################
# TWO-BODY KINEMATICS INTEGRATION
###############################################################################


def _integrate_twobody_cm_freegas(this, Ein, Eouts, order, mus, awr, kT, xs,
                                  mus_grid, wgts):
    """Integrates this distribution at a given incoming energy,
    over a given lab-frame mu range and given outgoing energy bounds
    for a target-in-motion.
    """

    # Generate the incoming distribution via interpolation
    if this._angle:
        adist = interpolate_distribution(this._angle, Ein)
    else:
        adist = Tabular(np.array([-1., 1.]), np.array([0.5, 0.5]),
                        'histogram')

    # If the angular distribution is ONLY tabular, then the freegas integrands
    # can be written more explicitly, giving much needed speed (4x) in free-gas
    # integration (note the same could be done for target-at-rest two-body, but
    # those integration routines are nearly instantaneous anyways and so the
    # extra lines provide no value)
    if isinstance(adist, Uniform):
        adist = Tabular(np.array([-1., 1.]), np.array([0.5, 0.5]),
                        'histogram')

    # For the same reasons as the above block, convert the string interpolation
    # keys to integers
    if adist._interpolation == 'histogram':
        adist_interp = 1
    elif adist._interpolation == 'linear-linear':
        adist_interp = 2
    elif adist._interpolation == 'linear-log':
        adist_interp = 3
    elif adist._interpolation == 'log-linear':
        adist_interp = 4
    elif adist._interpolation == 'log-log':
        adist_interp = 5

    # Set up results vector
    if order is not None:
        integral = np.zeros((len(Eouts) - 1, order + 1))
    else:
        integral = np.zeros((len(Eouts) - 1, len(mus) - 1))
    grid = np.empty(integral.shape[1], np.float)

    # Set the constants to be used in our calculation
    beta = (awr + 1.) / awr
    half_beta_2 = 0.25 * beta * beta
    alpha = awr / kT
    srch_args = (Ein, beta, alpha, awr, kT, half_beta_2, adist._x, adist._p,
                 adist_interp, xs._x, xs._y, xs._breakpoints,
                 xs._interpolation)

    # Find the Eout bounds
    # We will search both backward and forward scattering conditions to see
    # what the most extreme outgoing energies we need to consider are

    # Do the following for backward scattering
    # Now find the maximum value so we can bracket our range and also use it
    # to find our roots (when function passes tolerance * max)
    maximum = sopt.minimize_scalar(find_freegas_integral_max,
                                   bracket=[0., Ein + 12. / alpha],
                                   args=(_MU_BACK, *srch_args, _FGK_RESULT))
    Eout_peak = maximum.x
    func_peak = -maximum.fun

    # Now find the upper and lower roots around this peak
    Eout_min_back = \
        sopt.brentq(find_freegas_integral_root, Eouts[0], Eout_peak,
                    args=(_MU_BACK, *srch_args, _FGK_RESULT,
                          _FGK_ROOT_TOL * func_peak))
    Eout_max_back = \
        sopt.brentq(find_freegas_integral_root, Eout_peak, Eouts[-1],
                    args=(_MU_BACK, *srch_args, _FGK_RESULT,
                          _FGK_ROOT_TOL * func_peak))

    # Now we have to repeat for the top end;
    # Here we dont do mu=1, since the problem is unstable at mu=1, Ein=Eout,
    # which would give us a peak many orders of magnitude higher than the
    # surrounding points (so almost any point would pass the tolerance * peak
    # root check even with pretty low tolerances).
    # Instead we search at mu=0.9 and linearly interpolate to mu=1.
    maximum = \
        sopt.minimize_scalar(find_freegas_integral_max,
                             bracket=[0., Ein + 12. / alpha],
                             args=(_MU_FWD, *srch_args, _FGK_RESULT))
    Eout_peak = maximum.x
    func_peak = -maximum.fun
    Eout_min_fwd = sopt.brentq(find_freegas_integral_root,
                               Eouts[0], Eout_peak,
                               args=(_MU_FWD, *srch_args, _FGK_RESULT,
                                     _FGK_ROOT_TOL * func_peak))
    Eout_max_fwd = sopt.brentq(find_freegas_integral_root,
                               Eout_peak, Eouts[-1],
                               args=(_MU_FWD, *srch_args, _FGK_RESULT,
                                     _FGK_ROOT_TOL * func_peak))

    Eout_min_fwd = (1. - _INTERP_MU) * Eout_min_back + \
        _INTERP_MU * Eout_min_fwd
    Eout_max_fwd = (1. - _INTERP_MU) * Eout_max_back + \
        _INTERP_MU * Eout_max_fwd
    Eout_min = min(Eout_min_back, Eout_min_fwd)
    Eout_max = max(Eout_max_back, Eout_max_fwd)

    # Finally, perform the integration
    if order is not None:
        integrate_legendres_freegas(Ein, Eouts, beta, alpha, awr, kT,
                                    half_beta_2, adist._x, adist._p,
                                    adist_interp, xs._x, xs._y,
                                    xs._breakpoints, xs._interpolation,
                                    grid, integral, mus_grid, wgts, Eout_min,
                                    Eout_max)
    else:
        integrate_histogram_freegas(Ein, Eouts, mus, *args, grid, integral,
                                    Eout_min, Eout_max)
    return integral


def _integrate_twobody_cm_TAR(this, Ein, Eouts, awr, q_value, order, mus):
    """Integrates this distribution at a given incoming energy,
    over a given lab-frame mu range and given outgoing energy bounds when a
    center-of-mass distribution is provided assuming a target-at-rest
    """

    # Generate the incoming distribution via interpolation
    if this._angle:
        adist = interpolate_distribution(this._angle, Ein)
    else:
        adist = Uniform(-1., 1.)

    # Set up results vector and integrand function
    if order is not None:
        integral = np.zeros((len(Eouts) - 1, order + 1))
    else:
        integral = np.zeros((len(Eouts) - 1, len(mus) - 1))

    integrate_twobody_cm_TAR(Ein, Eouts, awr, q_value, adist, integral,
                             order is not None, mus)

    return integral


###############################################################################
# GENERIC CENTER-OF-MASS INTEGRATION
###############################################################################


def _integrate_generic_cm(this, Ein, Eouts, awr, order, mus, mus_grid, wgts):
    """Integrates a distribution at a given incoming energy, over a given
    outgoing energy bounds when Legendre coefficients are requested.
    """

    # Initialize the memory for the result.
    if order is not None:
        integral = np.zeros((len(Eouts) - 1, order + 1))
    else:
        integral = np.zeros((len(Eouts) - 1, len(mus) - 1))

    # Pre-process to obtain our energy and angle distributions
    if isinstance(this, UncorrelatedAngleEnergy):
        eadist = _preprocess_uncorr(this, Ein)
    elif isinstance(this, KalbachMann):
        eadist = _preprocess_kalbach(this, Ein)
    elif isinstance(this, CorrelatedAngleEnergy):
        eadist = _preprocess_corr(this, Ein)
    elif isinstance(this, NBodyPhaseSpace):
        eadist = _preprocess_nbody(this, Ein)

    # Call our integration routine
    Eout_max = eadist.Eout_max()
    integrate_cm(Ein, Eouts, Eout_max, awr, integral, eadist,
                 order is not None, mus)

    return integral


###############################################################################
# LAB-FRAME INTEGRATION
###############################################################################

def _integrate_generic_lab(this, Ein, Eouts, order, mus, mus_grid, wgts):
    """Integrates this distribution at a given incoming energy,
    over a given lab-frame mu range and given outgoing energy bounds when a
    lab-frame distribution is provided.
    """

    # Initialize the memory for the result.
    if order is not None:
        integral = np.zeros((len(Eouts) - 1, order + 1))
    else:
        integral = np.zeros((len(Eouts) - 1, len(mus) - 1))
    grid = np.empty(integral.shape[1], np.float)

    legendre = order is not None

    # Pre-process and call the distribution-type specific method
    if isinstance(this, UncorrelatedAngleEnergy):
        eadist = _preprocess_uncorr(this, Ein)

    elif isinstance(this, CorrelatedAngleEnergy):
        eadist = _preprocess_corr(this, Ein)

    elif isinstance(this, NBodyPhaseSpace):
        eadist = _preprocess_nbody(this, Ein)

    if legendre:
        eadist.integrate_lab_legendre(Eouts, integral, grid, mus_grid, wgts)
    else:
        eadist.integrate_lab_histogram(Eouts, integral, grid, mus)

    return integral
