import numpy as np

import openmc.stats
from openmc.data import KalbachMann, UncorrelatedAngleEnergy, \
    CorrelatedAngleEnergy, LevelInelastic, NBodyPhaseSpace, Uniform
from .interpolators import *
from .uncorrelated import Uncorrelated
from .correlated import Correlated
from .kalbach import KM
from .nbody import NBody
from .twobody_tar import TwoBody_TAR
from .twobody_fgk import TwoBody_FGK


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
        is in the center of mass frame (True) or not.
    awr : float
        Atomic weight ratio
    freegas_cutoff : float
        Maximal energy (in eV) in which the free-gas kernel is applied
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
                return integrate_twobody_cm_freegas(this, Ein, Eouts, order,
                                                    mus, awr, kT, xs,
                                                    mus_grid, wgts)
            else:
                return integrate_twobody_cm_TAR(this, Ein, Eouts, awr,
                                                q_value, order, mus)
    # Now move on to general center-of-mass or lab-frame integration
    return integrate_generic(this, Ein, Eouts, awr, order, mus, mus_grid,
                             wgts, cm)


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


def integrate_twobody_cm_freegas(this, Ein, Eouts, order, mus, awr, kT, xs,
                                 mus_grid, wgts):
    """Integrates this distribution at a given incoming energy,
    over a given lab-frame mu range and given outgoing energy bounds
    for a target-in-motion.
    """

    # Generate the incoming distribution via interpolation
    if this._angle:
        adist = interpolate_distribution(this._angle, Ein)
    else:
        adist = openmc.stats.Tabular(np.array([-1., 1.]), np.array([0.5, 0.5]),
                                     'histogram')

    # If the angular distribution is ONLY tabular, then the freegas integrands
    # can be written more explicitly, giving much needed speed (4x) in free-gas
    # integration (note the same could be done for target-at-rest two-body, but
    # those integration routines are nearly instantaneous anyways and so the
    # extra lines provide no value)
    if isinstance(adist, Uniform):
        adist = openmc.stats.Tabular(np.array([-1., 1.]), np.array([0.5, 0.5]),
                                     'histogram')

    eadist = TwoBody_FGK(adist, Ein, awr, kT, xs, Eouts)

    if order is not None:
        integral = eadist.integrate_cm_legendre(Ein, Eouts, order, mus_grid,
                                                wgts)
    else:
        integral = eadist.integrate_cm_histogram(Ein, Eouts, mus)

    return integral


def integrate_twobody_cm_TAR(this, Ein, Eouts, awr, q_value, order, mus):
    """Integrates this distribution at a given incoming energy,
    over a given lab-frame mu range and given outgoing energy bounds when a
    center-of-mass distribution is provided assuming a target-at-rest
    """

    # Generate the incoming distribution via interpolation
    if this._angle:
        adist = interpolate_distribution(this._angle, Ein)
    else:
        adist = openmc.stats.Tabular(np.array([-1., 1.]), np.array([0.5, 0.5]),
                                     'histogram')

    eadist = TwoBody_TAR(adist, Ein, awr, q_value)

    if order is not None:
        integral = eadist.integrate_cm_legendre(Ein, Eouts, order)
    else:
        integral = eadist.integrate_cm_histogram(Ein, Eouts, mus)

    return integral


###############################################################################
# GENERIC NON-TWO-BODY INTEGRATION
###############################################################################


def integrate_generic(this, Ein, Eouts, awr, order, mus, mus_grid, wgts, cm):
    """Integrates this distribution at a given incoming energy,
    over a given lab or cm-frame mu range and given outgoing energy bounds.
    """

    # Pre-process to obtain our energy and angle distributions
    if isinstance(this, UncorrelatedAngleEnergy):
        eadist = _preprocess_uncorr(this, Ein)

    elif isinstance(this, KalbachMann):
        eadist = _preprocess_kalbach(this, Ein)

    elif isinstance(this, CorrelatedAngleEnergy):
        eadist = _preprocess_corr(this, Ein)

    elif isinstance(this, NBodyPhaseSpace):
        eadist = _preprocess_nbody(this, Ein)

    # Initialize the memory for the result and call our integration routine
    # The specific calls depends on if we are using a Legendre expansion or not
    if cm:
        if order is not None:
            integral = eadist.integrate_cm_legendre(Ein, Eouts, awr, order)
        else:
            integral = eadist.integrate_cm_histogram(Ein, Eouts, awr, mus)
    else:
        if order is not None:
            integral = eadist.integrate_lab_legendre(Eouts, order, mus_grid,
                                                     wgts)
        else:
            integral = eadist.integrate_lab_histogram(Eouts, mus)

    return integral
