from numbers import Real

import numpy as np
from scipy.special import erf

import openmc.checkvalue as cv
import openmc.data
import openmc.stats


# Linearization tolerance for EnergyDistribution objects
_LIN_TOL = 0.00001


def interpolate_kalbach(this, Ein):
    """Interpolates the data an openmc.data.KalbachMann object and returns
    the outgoing energy PDF, precompound and slope factors as a function of the
    outgoing energy, all at the specified Ein.
    """
    if Ein >= this._energy[-1]:
        i = len(this._energy) - 2
    elif Ein <= this._energy[0]:
        i = 0
    else:
        i = np.searchsorted(this._energy, Ein) - 1

    edist = \
        interpolate_function(this._energy_out[i], this._energy_out[i + 1],
                             Ein, this._energy[i], this.energy[i + 1],
                             interp_type='unit-base')
    precompound = \
        interpolate_function(this._precompound[i],
                             this._precompound[i + 1], Ein,
                             this._energy[i], this.energy[i + 1],
                             interp_type='unit-base')
    slope = \
        interpolate_function(this._slope[i], this._slope[i + 1],
                             Ein, this._energy[i], this.energy[i + 1],
                             interp_type='unit-base')

    return edist, precompound, slope


def interpolate_uncorr(this, Ein):
    """Interpolates the data an openmc.data.UncorrelatedAngleEnergy object and
    returns the outgoing energy PDF and angular distribution at the specified
    Ein.
    """
    edist = interpolate_distribution(this._energy, Ein)

    if this.angle:
        adist = interpolate_distribution(this._angle, Ein)
    else:
        adist = openmc.stats.Uniform(-1., 1.)

    return edist, adist


def interpolate_corr(this, Ein):
    """Interpolates the data an openmc.data.CorrelatedAngleEnergy object and
    returns the outgoing energy PDF and angular distributions at the specified
    Ein.
    """
    if Ein >= this._energy[-1]:
        i = len(this._energy) - 2
    elif Ein <= this._energy[0]:
        i = 0
    else:
        i = np.searchsorted(this._energy, Ein) - 1
    edist = interpolate_function(this._energy_out[i],
                                 this._energy_out[i + 1], Ein,
                                 this._energy[i], this._energy[i + 1],
                                 'unit-base')

    domain_low = this._energy_out[i].get_domain(Ein)
    domain_high = this._energy_out[i + 1].get_domain(Ein)
    domain_interp = edist.get_domain(Ein)
    dE_low = domain_low[1] - domain_low[0]
    dE_high = domain_high[1] - domain_low[0]
    dE_interp = domain_interp[1] - domain_interp[0]

    adists = []
    for eo, Eout in enumerate(edist._x):
        x = (Eout - domain_interp[0]) / dE_interp
        # Now find what the corresponding outgoing energy would be in
        # the low grid for its own value of x
        Eout_low = x * dE_low + domain_low[0]

        # Now find the angular distribution at this Eout on the low grid
        if Eout_low >= domain_low[1]:
            j = len(this._energy_out[i]._x) - 2
        elif Eout_low <= domain_low[0]:
            j = 0
        else:
            j = np.searchsorted(this._energy_out[i]._x, Eout_low) - 1

        adist_low = \
            interpolate_function(this._mu[i][j], this._mu[i][j + 1],
                                 Eout_low, this._energy_out[i]._x[j],
                                 this._energy_out[i]._x[j + 1], 'unit-base')

        # Do the same for the high point
        Eout_high = x * dE_high + domain_high[0]
        if Eout_high >= domain_high[1]:
            j = len(this._energy_out[i + 1]._x) - 2
        elif Eout_high <= domain_high[0]:
            j = 0
        else:
            j = np.searchsorted(this._energy_out[i + 1]._x, Eout_high) - 1
        adist_high = \
            interpolate_function(this._mu[i + 1][j],
                                 this._mu[i + 1][j + 1],
                                 Eout_high, this._energy_out[i + 1]._x[j],
                                 this._energy_out[i + 1]._x[j + 1],
                                 'unit-base')

        # We now have an angular distribution from the low and high
        # distributions, lets just interpolate between the two to find our
        # current Ein, Eout pair's angular distribution
        adist = interpolate_function(adist_low, adist_high, Ein,
                                     this._energy[i], this._energy[i + 1],
                                     'unit-base')
        adists.append(adist)

    return edist, adists


def interpolate_distribution(this, Ein):
    """Interpolate on an energy or angle distribution given the incoming energy

    Parameters
    ----------
    Ein : Real
        Incoming energy

    Returns
    -------
        openmc.data.EnergyDistribution, or openmc.data.AngleDistribution
        Interpolated distribution with interpolated data

    """

    cv.check_type('Ein', Ein, Real)
    cv.check_greater_than('Ein', Ein, 0, equality=True)

    if isinstance(this, openmc.data.AngleDistribution):
        return _interpolate_AngleDistribution(this, Ein)
    elif isinstance(this, openmc.data.ContinuousTabular):
        return _interpolate_ContinuousTabularDistribution(this, Ein)
    elif isinstance(this, openmc.data.ArbitraryTabulated):
        return _interpolate_ArbitraryTabulatedDistribution(this, Ein)
    elif isinstance(this, openmc.data.Evaporation):
        return _interpolate_EvaporationDistribution(this, Ein)
    elif isinstance(this, openmc.data.GeneralEvaporation):
        return _interpolate_GeneralEvaporationDistribution(this, Ein)
    elif isinstance(this, openmc.data.MaxwellEnergy):
        return _interpolate_MaxwellDistribution(this, Ein)
    elif isinstance(this, openmc.data.WattEnergy):
        return _interpolate_WattDistribution
    else:
        raise ValueError("Invalid class provided: " + str(type(this)))


def interpolate_function(this, that, my_in, this_in, that_in, interp_type):
    """Interpolates a new object made up of this and that at a point my_in
    where this and that come from this_in and that_in, respectively.

    Parameters
    ----------
    this : {openmc.data.Function1D, or openmc.stats.Univariate}
        Bottom data point to interpolate to.
    that : {openmc.data.Function1D, or openmc.stats.Univariate}
        Other data point to interpolate to.
    my_in : float
        Value to interpolate to
    this_in : float
        Value of this on the interpolating grid
    that_in : float
        Value of that on the interpolating grid
    interp_type : {'unit-base', 'nearest', 'linear'}:
        Which type of interpolating to perform: either unit-base, nearest
        point, or linear interpolation.

    Returns
    -------
    {openmc.data.Function1D, or openmc.stats.Univariate}
        New object which resulted from the interpolation. The resultant type
        will be the same as the types of this and that, unless this and that
        are provided as openmc.stats.Uniform objects; these will be returned
        as openmc.stats.Tabular objects since inerpolation of a uniform grid
        requires representation with a non-uniform format.

    """

    cv.check_type('that', that, type(this))
    cv.check_type('my_in', my_in, Real)
    cv.check_type('this_in', this_in, Real)
    cv.check_type('that_in', that_in, Real)
    cv.check_value('interpolation type', interp_type,
                   ['unit-base', 'nearest', 'linear'])

    if isinstance(this, openmc.data.Tabulated1D):
        return _interpolate_Tabulated1D(this, that, my_in, this_in, that_in,
                                        interp_type)
    elif isinstance(this, openmc.data.Polynomial):
        return _interpolate_Polynomial(this, that, my_in, this_in, that_in,
                                       interp_type)
    elif isinstance(this, openmc.stats.Discrete):
        return _interpolate_Discrete(this, that, my_in, this_in, that_in,
                                     interp_type)
    elif isinstance(this, openmc.stats.Uniform):
        return _interpolate_Uniform(this, that, my_in, this_in, that_in,
                                    interp_type)
    elif isinstance(this, openmc.stats.Maxwell):
        return _interpolate_Maxwell(this, that, my_in, this_in, that_in,
                                    interp_type)
    elif isinstance(this, openmc.stats.Watt):
        return _interpolate_Watt(this, that, my_in, this_in, that_in,
                                 interp_type)
    elif isinstance(this, openmc.stats.Tabular):
        return _interpolate_Tabular(this, that, my_in, this_in, that_in,
                                    interp_type)
    elif isinstance(this, openmc.stats.Legendre):
        return _interpolate_Legendre(this, that, my_in, this_in, that_in,
                                     interp_type)
    elif isinstance(this, openmc.stats.Mixture):
        return _interpolate_Mixture(this, that, my_in, this_in, that_in,
                                    interp_type)
    elif this is None:
        # Then we have an isotropic distribution
        return openmc.stats.Tabular(np.array([-1., 1.]), np.array([.5, .5]),
                                    'histogram', True)
    else:
        # In addition to unforeseen configurations, it is also possible
        # that this is a None, but that is not (or vice versa); this would also
        # put us here. If that shows up, address it.
        raise ValueError("Invalid class provided!", type(this))


def _interpolate_AngleDistribution(this, Ein):
    if Ein >= this._energy[-1]:
        i = len(this._energy) - 2
    elif Ein <= this._energy[0]:
        i = 0
    else:
        i = np.searchsorted(this._energy, Ein) - 1

    return interpolate_function(this._mu[i], this._mu[i + 1], Ein,
                                this._energy[i], this._energy[i + 1],
                                interp_type='unit-base')


def _interpolate_ContinuousTabularDistribution(this, Ein):
    if Ein >= this._energy[-1]:
        i = len(this._energy) - 2
    elif Ein <= this._energy[0]:
        i = 0
    else:
        i = np.searchsorted(this._energy, Ein) - 1

    # Read number of interpolation regions and incoming energies
    if len(this._interpolation) == 1:
        histogram_interp = (this._interpolation[0] == 1)
    else:
        histogram_interp = False

    # If histogram interpolation, we will just be using the i-th bin
    if histogram_interp:
        # Then just take our current point, but convert to a Cythonized object
        if isinstance(this._energy_out[i], openmc.stats.Discrete):
            return this._energy_out[i]
        elif isinstance(this, openmc.stats.Uniform):
            return this._energy_out[i]
        elif isinstance(this, openmc.stats.Tabular):
            return this._energy_out[i]
        elif isinstance(this, openmc.stats.Legendre):
            return this._energy_out[i]
    else:
        return interpolate_function(this._energy_out[i],
                                    this._energy_out[i + 1], Ein,
                                    this._energy[i], this._energy[i + 1],
                                    interp_type='unit-base')


def _interpolate_ArbitraryTabulatedDistribution(this, Ein):
    if Ein >= this._energy[-1]:
        i = len(this._energy) - 2
    elif Ein <= this._energy[0]:
        i = 0
    else:
        i = np.searchsorted(this._energy, Ein) - 1

    new = interpolate_function(this._pdf[i], this._pdf[i + 1], Ein,
                               this._energy[i], this._energy[i + 1],
                               interp_type='unit_base')

    return new


def _interpolate_GeneralEvaporationDistribution(this, Ein):
    # Get the right value of theta
    theta = this.theta(Ein)
    x_guess = np.array(this.get_domain(Ein))
    if x_guess[0] == 0:
        x_guess[0] = 1.e-5

    def func(Eout):
        x_val = Eout / theta
        return this.g(x_val)

    x, p = openmc.data.linearize(x_guess, func, tolerance=_LIN_TOL)
    new = openmc.stats.Tabular(x, p)

    return new


def _interpolate_MaxwellDistribution(this, Ein):
    # Get the right value of theta
    theta = this.theta(Ein)
    u = this.u
    EmU_th = (Ein - u) / theta
    norm = theta ** 1.5 * (0.5 * np.sqrt(np.pi) * erf(np.sqrt(EmU_th)) -
                           np.sqrt(EmU_th) * np.exp(-EmU_th))
    x_guess = np.array(this.get_domain(Ein))
    if x_guess[0] == 0:
        x_guess[0] = 1.e-5

    def func(Eout):
        return np.sqrt(Eout) / norm * np.exp(-Eout / theta)

    x, p = openmc.data.linearize(x_guess, func, tolerance=_LIN_TOL)
    new = openmc.stats.Tabular(x, p)

    return new


def _interpolate_EvaporationDistribution(this, Ein):
    # Get the right value of theta
    theta = this.theta(Ein)
    u = this.u
    EmU_th = (Ein - u) / theta
    norm = theta * theta * (1. - np.exp(-EmU_th) * (1. + EmU_th))
    x_guess = np.array(this.get_domain(Ein))
    if x_guess[0] == 0:
        x_guess[0] = 1.e-5

    def func(Eout):
        return Eout / norm * np.exp(-Eout / theta)

    x, p = openmc.data.linearize(x_guess, func, tolerance=_LIN_TOL)
    new = openmc.stats.Tabular(x, p)

    return new


def _interpolate_WattDistribution(this, Ein):
    # Get the right value of theta
    a = this.a(Ein)
    b = this.b(Ein)
    u = this.u

    EmU_a = (Ein - u) / a
    root_EmU_a = np.sqrt(EmU_a)
    ab_4 = 0.25 * a * b
    root_ab_4 = np.sqrt(ab_4)
    norm = 0.5 * np.sqrt(np.pi * ab_4) * np.exp(ab_4) * \
        (erf(root_EmU_a - root_ab_4) + erf(root_EmU_a + root_ab_4)) - \
        a * np.exp(-EmU_a) * np.sinh(np.sqrt(b * (Ein - u)))

    x_guess = np.array(this.get_domain(Ein))
    if x_guess[0] == 0:
        x_guess[0] = 1.e-5

    def func(Eout):
        return np.exp(-Eout / a) / norm * np.sinh(np.sqrt(b * Eout))

    x, p = openmc.data.linearize(x_guess, func, tolerance=_LIN_TOL)
    new = openmc.stats.Tabular(x, p)

    return new


def _interpolate_Tabulated1D(this, that, my_in, this_in, that_in, interp_type):

    if interp_type == 'nearest':
        if np.abs(my_in - this_in) <= np.abs(my_in - that_in):
            x, y = this._x, this._y
        else:
            x, y = that._x, that._y
    elif interp_type == 'linear':
        f = (my_in - this_in) / (that_in - this_in)
        if np.array_equal(this._x, that._x):
            x = this._x
            y = (1. - f) * this._y + f * that._y

        else:
            x = np.union1d(this._x, that._x)
            y = (1. - f) * this(x) + f * that(x)
    elif interp_type == 'unit-base':
        # The first step is to convert this and that to a 'unit-base',
        # that is, an x,y grid whose domain is [0,1] instead of its current
        # domain.
        # First, get a common x grid structure between this and that
        this_dx = this._x[-1] - this._x[0]
        if this_dx > 0.:
            inv_dx = 1. / this_dx
        else:
            inv_dx = 0.
        ub_this_x = inv_dx * (this._x - this._x[0])

        that_dx = that._x[-1] - that._x[0]
        if that_dx > 0.:
            inv_dx = 1. / that_dx
        else:
            inv_dx = 0.
        ub_that_x = inv_dx * (that._x - that._x[0])

        ub_x = np.union1d(ub_this_x, ub_that_x)
        ub_x_this = this._x[0] + ub_x * this_dx
        ub_x_that = that._x[0] + ub_x * that_dx

        # Get our interpolant
        f = (my_in - this_in) / (that_in - this_in)
        y = (1. - f) * this(ub_x_this) + f * that(ub_x_that)

        # Now get a new energy grid
        x = (1. - f) * ub_x_this + f * ub_x_that

    # Assume we can use breakpoints and interpolation from this
    new = openmc.data.Tabulated1D(x, y, this._breakpoints, this._interpolation)
    return new


def _interpolate_Polynomial(this, that, my_in, this_in, that_in, interp_type):

    if interp_type == 'nearest':
        if np.abs(my_in - this_in) <= np.abs(my_in - that_in):
            return this
        else:
            return that
    elif interp_type == 'unit-base':
        # The first step is to convert this and that to a 'unit-base',
        # that is, an x,y grid whose domain is [0,1] instead of its current
        # domain.

        # With polynomials this is easy
        ub_this = this.convert(domain=[0, 1])
        ub_that = that.convert(domain=[0, 1])

        # Get our interpolant
        f = (my_in - this_in) / (that_in - this_in)
        ub = np.polynomial.polynomial.polyadd((1. - f) * ub_this, f * ub_that)
        x_lo = ((1. - f) * this.domain[0] + f * that.domain[0])
        x_hi = ((1. - f) * this.domain[1] + (this.domain[1] - this.domain[0]) +
                f * that.domain[1] + (that.domain[1] - that.domain[0]))
        new = ub.convert(domain=[x_lo, x_hi])

        return new


def _interpolate_Discrete(this, that, my_in, this_in, that_in, interp_type):

    if interp_type == 'nearest':
        if np.abs(my_in - this_in) <= np.abs(my_in - that_in):
            x, p = this._x, this._p
        else:
            x, p = that._x, that._p

    elif interp_type == 'linear':
        f = (my_in - this_in) / (that_in - this_in)
        if np.array_equal(this._x, that._x):
            # Same x's so just interpolate the p's
            x = this._x
            p = (1. - f) * this._p + f * that._p
        elif len(this._x) == len(that._x):
            # Same number of x's so we just have a shift in x we have to
            # include
            x = (1. - f) * this._x + f * that._x
            p = (1. - f) * this._p + f * that._p
        else:
            # different number of x's
            x = np.union1d(this._x, that._x)
            p = (1. - f) * this(x) + f * that(x)

    elif interp_type == 'unit-base':
        # The first step is to convert this and that to a 'unit-base',
        # that is, an x,y grid whose domain is [0,1] instead of its current
        # domain.
        # First, get a common x grid structure between this and that
        this_dx = this._x[-1] - this._x[0]
        if this_dx > 0.:
            inv_dx = 1. / this_dx
        else:
            inv_dx = 0.
        ub_this_x = inv_dx * (this._x - this._x[0])

        that_dx = that._x[-1] - that._x[0]
        if that_dx > 0.:
            inv_dx = 1. / that_dx
        else:
            inv_dx = 0.
        ub_that_x = inv_dx * (that._x - that._x[0])

        ub_x = np.union1d(ub_this_x, ub_that_x)

        # Get our interpolant
        f = (my_in - this_in) / (that_in - this_in)
        p = (1. - f) * this(ub_x) + f * that(ub_x)
        # Now get a new x grid
        x = (1. - f) * (this._x[0] + this_dx * ub_x) + \
            f * (that._x[0] + that_dx * ub_x)

    # Normalize p
    p = np.divide(p, np.sum(p))

    # And create our new distribution
    new = openmc.stats.Discrete(x, p)

    return new


def _interpolate_Uniform(this, that, my_in, this_in, that_in, interp_type):

    tab_this = _uniform_to_tabular(this)
    tab_that = _uniform_to_tabular(that)

    if interp_type == 'nearest':
        if np.abs(my_in - this_in) <= np.abs(my_in - that_in):
            new = tab_this
        else:
            new = tab_that
    else:
        new = _interpolate_Tabular(tab_this, tab_that, my_in, this_in,
                                   that_in, interp_type)

    return new


def _interpolate_Maxwell(this, that, my_in, this_in, that_in, interp_type):

    if interp_type == 'nearest':
        if np.abs(my_in - this_in) <= np.abs(my_in - that_in):
            return this
        else:
            return that
    elif interp_type in ['unit-base', 'linear']:
        # Interpolating a Maxwell is simply a matter of interpolating on
        # theta
        # Get our interpolant
        f = (my_in - this_in) / (that_in - this_in)
        theta = (1. - f) * this.theta + f * that.theta
        new = openmc.stats.Maxwell(theta)
        return new


def _interpolate_Watt(this, that, my_in, this_in, that_in, interp_type):

    if interp_type == 'nearest':
        if np.abs(my_in - this_in) <= np.abs(my_in - that_in):
            return this
        else:
            return that
    elif interp_type in ['unit-base', 'linear']:
        # Interpolating a Watt is simply a matter of interpolating on
        # a and b
        # Get our interpolant
        f = (my_in - this_in) / (that_in - this_in)
        a = (1. - f) * this.a + f * that.a
        b = (1. - f) * this.b + f * that.b
        new = openmc.stats.Watt(a, b)
        return new


def _interpolate_Tabular(this, that, my_in, this_in, that_in, interp_type):

    if interp_type == 'nearest':
        if np.abs(my_in - this_in) <= np.abs(my_in - that_in):
            x, p = this._x, this._p
        else:
            x, p = that._x, that._p

    elif interp_type == 'linear':
        f = (my_in - this_in) / (that_in - this_in)
        if np.array_equal(this._x, that._x):
            x = this._x
            p = (1. - f) * this._p + f * that._p

        else:
            x = np.union1d(this._x, that._x)
            p = (1. - f) * this(x) + f * that(x)

    elif interp_type == 'unit-base':
        # The first step is to convert this and that to a 'unit-base',
        # that is, an x grid whose domain is [0,1] instead of its current
        # domain.
        # First, get a common x grid structure between this and that
        this_dx = this._x[-1] - this._x[0]
        if this_dx > 0.:
            inv_dx = 1. / this_dx
        else:
            inv_dx = 0.
        ub_this_x = inv_dx * (this._x - this._x[0])

        that_dx = that._x[-1] - that._x[0]
        if that_dx > 0.:
            inv_dx = 1. / that_dx
        else:
            inv_dx = 0.
        ub_that_x = inv_dx * (that._x - that._x[0])

        ub_x = np.union1d(ub_this_x, ub_that_x)
        ub_x_this = this._x[0] + ub_x * this_dx
        ub_x_that = that._x[0] + ub_x * that_dx

        # Get our interpolant
        if that_in - this_in > 0.:
            f = (my_in - this_in) / (that_in - this_in)
        else:
            f = 0.

        p = (1. - f) * this(ub_x_this) + f * that(ub_x_that)

        # Now get a new x grid
        x = (1. - f) * ub_x_this + f * ub_x_that

    # Make sure any negative values of p from f.p. error are set to 0.
    p[p < 0] = 0.

    # Make sure that p is normalized still
    p = np.divide(p, np.trapz(p, x))
    new = openmc.stats.Tabular(x, p, this._interpolation)

    return new


def _interpolate_Legendre(this, that, my_in, this_in, that_in, interp_type):

    if interp_type == 'nearest':
        if np.abs(my_in - this_in) <= np.abs(my_in - that_in):
            return this
        else:
            return that
    elif interp_type in ['linear', 'unit-base']:
        # Find the interpolant
        f = (my_in - this_in) / (that_in - this_in)
        new = np.polynomial.polyadd((1. - f) * this._legendre_polynomial,
                                    f * that._legendre_polynomial)
        return new


def _interpolate_Mixture(this, that, my_in, this_in, that_in, interp_type):

        if interp_type == 'nearest':
            if np.abs(my_in - this_in) <= np.abs(my_in - that_in):
                return this
            else:
                return that
        elif interp_type in ['unit-base', 'linear']:
            # Find the interpolant
            f = (my_in - this_in) / (that_in - this_in)
            if len(this.probability) == len(that.probability):
                prob = np.zeros_like(this.probability)
                dist = [] * len(this.probability)
                for i in range(len(this.probability)):
                    prob[i] = (1. - f) * this.probability[i] + \
                        f * that.probability[i]
                    dist[i] = \
                        this.distribution[i].interpolate(that.distribution[i],
                                                         my_in, this_in,
                                                         that_in)
                new = openmc.stats.Mixture(prob, dist)
                return new
            else:
                raise ValueError("Can not interpolate on openmc.stats.Mixture "
                                 "objects with different number of "
                                 "distributions")


def _uniform_to_tabular(this):
    prob = 1. / (this._b - this._a)
    t = openmc.stats.Tabular(np.array([this._a, this._b]),
                             np.array([prob, prob]), 'histogram')
    return t
