from abc import ABCMeta, abstractmethod
from collections import Iterable
from numbers import Real
from xml.etree import ElementTree as ET

from six import add_metaclass
import numpy as np

import openmc.checkvalue as cv
from openmc.mixin import EqualityMixin
try:
    from .univariate_methods_cython import *
except ImportError:
    from .univariate_methods import *


_INTERPOLATION_SCHEMES = ['histogram', 'linear-linear', 'linear-log',
                          'log-linear', 'log-log']


@add_metaclass(ABCMeta)
class Univariate(EqualityMixin):
    """Probability distribution of a single random variable.

    The Univariate class is an abstract class that can be derived to implement a
    specific probability distribution.

    """

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, x):
        return NotImplementedError

    @abstractmethod
    def to_xml_element(self, element_name):
        return ''

    @abstractmethod
    def __len__(self):
        return 0

    def get_domain(self, Ein=None):
        return self._x[0], self._x[-1]

    @abstractmethod
    def integrate(self, lo, hi):
        pass


class Discrete(Univariate):
    """Distribution characterized by a probability mass function.

    The Discrete distribution assigns probability values to discrete values of a
    random variable, rather than expressing the distribution as a continuous
    random variable.

    Parameters
    ----------
    x : Iterable of Real
        Values of the random variable
    p : Iterable of Real
        Discrete probability for each value

    Attributes
    ----------
    x : Iterable of Real
        Values of the random variable
    p : Iterable of Real
        Discrete probability for each value

    """

    def __init__(self, x, p):
        super(Discrete, self).__init__()
        self.x = x
        self.p = p

        # Use the first unique entries (this overcomes a problem with
        # Cm-245, for example, when there are two Eout in a row with the same
        # value for an outgoing energy distribution in ENDF/B-VII.0)
        self.x, uniques = np.unique(self.x, return_index=True)
        self.p = self.p[uniques]

    def __len__(self):
        return len(self.x)

    def __call__(self, x):
        # Handle both array and scalar input
        if isinstance(x, Iterable):
            return np.fromiter((discrete_call(self, x_i) for x_i in x),
                               np.float, len(x))
        else:
            return discrete_call(self, x)

    @property
    def x(self):
        return self._x

    @property
    def p(self):
        return self._p

    @x.setter
    def x(self, x):
        if isinstance(x, Real):
            x = [x]
        cv.check_type('discrete values', x, Iterable, Real)
        self._x = x

    @p.setter
    def p(self, p):
        if isinstance(p, Real):
            p = [p]
        cv.check_type('discrete probabilities', p, Iterable, Real)
        for pk in p:
            cv.check_greater_than('discrete probability', pk, 0.0, True)
        self._p = p

    def _eval(self, x):
        """The same as __call__ without the overhead"""
        return discrete_call(self, x)

    def integrate(self, lo, hi):
        cv.check_type('lo', lo, Real)
        cv.check_type('hi', hi, Real)
        return discrete_integrate(self, lo, hi)

    def to_xml_element(self, element_name):
        """Return XML representation of the discrete distribution

        Parameters
        ----------
        element_name : str
            XML element name

        Returns
        -------
        element : xml.etree.ElementTree.Element
            XML element containing discrete distribution data

        """
        element = ET.Element(element_name)
        element.set("type", "discrete")

        params = ET.SubElement(element, "parameters")
        params.text = ' '.join(map(str, self.x)) + ' ' + \
            ' '.join(map(str, self.p))

        return element


class Uniform(Univariate):
    """Distribution with constant probability over a finite interval [a,b]

    Parameters
    ----------
    a : Real, optional
        Lower bound of the sampling interval. Defaults to zero.
    b : Real, optional
        Upper bound of the sampling interval. Defaults to unity.

    Attributes
    ----------
    a : Real
        Lower bound of the sampling interval
    b : Real
        Upper bound of the sampling interval

    """

    def __init__(self, a=0.0, b=1.0):
        super(Uniform, self).__init__()
        self.a = a
        self.b = b
        cv.check_greater_than('b > a', b, minimum=a)

    def __call__(self, x):
        # Handle both array and scalar input
        if isinstance(x, Iterable):
            return np.fromiter((uniform_call(self, x_i) for x_i in x),
                               np.float, len(x))
        else:
            return uniform_call(self, x)

    def __len__(self):
        return 2

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @a.setter
    def a(self, a):
        cv.check_type('Uniform a', a, Real)
        self._a = a

    @b.setter
    def b(self, b):
        cv.check_type('Uniform b', b, Real)
        self._b = b

    def _eval(self, x):
        """The same as __call__ without the overhead"""
        return uniform_call(self, x)

    def get_domain(self, Ein=None):
        return self._a, self._b

    def integrate(self, lo, hi):
        cv.check_type('lo', lo, Real)
        cv.check_type('hi', hi, Real)
        return uniform_integrate(self, lo, hi)

    def to_xml_element(self, element_name):
        """Return XML representation of the uniform distribution

        Parameters
        ----------
        element_name : str
            XML element name

        Returns
        -------
        element : xml.etree.ElementTree.Element
            XML element containing uniform distribution data

        """
        element = ET.Element(element_name)
        element.set("type", "uniform")
        element.set("parameters", '{} {}'.format(self.a, self.b))
        return element


class Maxwell(Univariate):
    """Maxwellian distribution in energy.

    The Maxwellian distribution in energy is characterized by a single parameter
    :math:`\theta` and has a density function :math:`p(E) dE = c E e^{-E/\theta}
    dE`.

    Parameters
    ----------
    theta : Real
        Effective temperature for distribution

    Attributes
    ----------
    theta : Real
        Effective temperature for distribution

    """

    def __init__(self, theta):
        super(Maxwell, self).__init__()
        self.theta = theta

    def __call__(self, x):
        # Handle both array and scalar input
        if isinstance(x, Iterable):
            return np.fromiter((maxwell_call(self, x_i) for x_i in x),
                               np.float, len(x))
        else:
            return maxwell_call(self, x)

    def __len__(self):
        return 1

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        cv.check_type('Maxwell temperature', theta, Real)
        cv.check_greater_than('Maxwell temperature', theta, 0.0)
        self._theta = theta

    def _eval(self, x):
        """The same as __call__ without the overhead"""
        return maxwell_call(self, x)

    def get_domain(self, Ein=None):
        return 0., np.inf

    def to_xml_element(self, element_name):
        """Return XML representation of the Maxwellian distribution

        Parameters
        ----------
        element_name : str
            XML element name

        Returns
        -------
        element : xml.etree.ElementTree.Element
            XML element containing Maxwellian distribution data

        """
        element = ET.Element(element_name)
        element.set("type", "maxwell")
        element.set("parameters", str(self.theta))
        return element


class Watt(Univariate):
    r"""Watt fission energy spectrum.

    The Watt fission energy spectrum is characterized by two parameters
    :math:`a` and :math:`b` and has density function :math:`p(E) dE = c e^{-E/a}
    \sinh \sqrt{b \, E} dE`.

    Parameters
    ----------
    a : Real
        First parameter of distribution
    b : Real
        Second parameter of distribution

    Attributes
    ----------
    a : Real
        First parameter of distribution
    b : Real
        Second parameter of distribution

    """

    def __init__(self, a=0.988e6, b=2.249e-6):
        super(Watt, self).__init__()
        self.a = a
        self.b = b

    def __call__(self, x):
        # Handle both array and scalar input
        if isinstance(x, Iterable):
            return np.fromiter((watt_call(self, x_i) for x_i in x),
                               np.float, len(x))
        else:
            return watt_call(self, x)

    def __len__(self):
        return 2

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @a.setter
    def a(self, a):
        cv.check_type('Watt a', a, Real)
        cv.check_greater_than('Watt a', a, 0.0)
        self._a = a

    @b.setter
    def b(self, b):
        cv.check_type('Watt b', b, Real)
        cv.check_greater_than('Watt b', b, 0.0)
        self._b = b

    def _eval(self, x):
        """The same as __call__ without the overhead"""
        return watt_call(self, x)

    def get_domain(self, Ein=None):
        return 0., np.inf

    def to_xml_element(self, element_name):
        """Return XML representation of the Watt distribution

        Parameters
        ----------
        element_name : str
            XML element name

        Returns
        -------
        element : xml.etree.ElementTree.Element
            XML element containing Watt distribution data

        """
        element = ET.Element(element_name)
        element.set("type", "watt")
        element.set("parameters", '{} {}'.format(self.a, self.b))
        return element


class Tabular(Univariate):
    """Piecewise continuous probability distribution.

    This class is used to represent a probability distribution whose density
    function is tabulated at specific values with a specified interpolation
    scheme.

    Parameters
    ----------
    x : Iterable of Real
        Tabulated values of the random variable
    p : Iterable of Real
        Tabulated probabilities
    interpolation : {'histogram', 'linear-linear', 'linear-log', 'log-linear', 'log-log'}, optional
        Indicate whether the density function is constant between tabulated
        points or linearly-interpolated. Defaults to 'linear-linear'.
    ignore_negative : bool
        Ignore negative probabilities

    Attributes
    ----------
    x : Iterable of Real
        Tabulated values of the random variable
    p : Iterable of Real
        Tabulated probabilities
    interpolation : {'histogram', 'linear-linear', 'linear-log', 'log-linear', 'log-log'}, optional
        Indicate whether the density function is constant between tabulated
        points or linearly-interpolated.

    """

    def __init__(self, x, p, interpolation='linear-linear',
                 ignore_negative=False):
        super(Tabular, self).__init__()
        self._ignore_negative = ignore_negative
        self.x = x
        self.p = p
        self.interpolation = interpolation

        # Use the first unique entries (this overcomes a problem with
        # Cm-245, for example, when there are two Eout in a row with the same
        # value for an outgoing energy distribution in ENDF/B-VII.0)
        self.x, uniques = np.unique(self.x, return_index=True)
        self.p = self.p[uniques]

    def __call__(self, x):
        # Handle both array and scalar input
        if isinstance(x, Iterable):
            return np.fromiter((tabular_call(self, x_i) for x_i in x),
                               np.float, len(x))
        else:
            return tabular_call(self, x)

    def __len__(self):
        return len(self.x)

    @property
    def x(self):
        return self._x

    @property
    def p(self):
        return self._p

    @property
    def interpolation(self):
        return self._interpolation

    @x.setter
    def x(self, x):
        cv.check_type('tabulated values', x, Iterable, Real)
        self._x = x

    @p.setter
    def p(self, p):
        cv.check_type('tabulated probabilities', p, Iterable, Real)
        if not self._ignore_negative:
            for pk in p:
                cv.check_greater_than('tabulated probability', pk, 0.0, True)
        self._p = p

    @interpolation.setter
    def interpolation(self, interpolation):
        cv.check_value('interpolation', interpolation, _INTERPOLATION_SCHEMES)
        self._interpolation = interpolation

    def _eval(self, x):
        """The same as __call__ without the overhead"""
        return tabular_call(self, x)

    def integrate(self, lo, hi):
        cv.check_type('lo', lo, Real)
        cv.check_type('hi', hi, Real)
        return tabular_integrate(self, lo, hi)

    def to_xml_element(self, element_name):
        """Return XML representation of the tabular distribution

        Parameters
        ----------
        element_name : str
            XML element name

        Returns
        -------
        element : xml.etree.ElementTree.Element
            XML element containing tabular distribution data

        """
        element = ET.Element(element_name)
        element.set("type", "tabular")
        element.set("interpolation", self.interpolation)

        params = ET.SubElement(element, "parameters")
        params.text = ' '.join(map(str, self.x)) + ' ' + ' '.join(map(str, self.p))

        return element


class Legendre(Univariate):
    r"""Probability density given by a Legendre polynomial expansion
    :math:`\sum\limits_{\ell=0}^N \frac{2\ell + 1}{2} a_\ell P_\ell(\mu)`.

    Parameters
    ----------
    coefficients : Iterable of Real
        Expansion coefficients :math:`a_\ell`. Note that the :math:`(2\ell +
        1)/2` factor should not be included.

    Attributes
    ----------
    coefficients : Iterable of Real
        Expansion coefficients :math:`a_\ell`. Note that the :math:`(2\ell +
        1)/2` factor should not be included.

    """

    def __init__(self, coefficients):
        self.coefficients = coefficients

    def __call__(self, x):
        return self._legendre_polynomial(x)

    def __len__(self):
        return len(self._legendre_polynomial.coef)

    @property
    def coefficients(self):
        poly = self._legendre_polynomial
        l = np.arange(poly.degree() + 1)
        return 2./(2.*l + 1.) * poly.coef

    @coefficients.setter
    def coefficients(self, coefficients):
        cv.check_type('Legendre expansion coefficients', coefficients,
                      Iterable, Real)
        for l in range(len(coefficients)):
            coefficients[l] *= (2.*l + 1.)/2.
        self._legendre_polynomial = np.polynomial.legendre.Legendre(
            coefficients)

    def _eval(self, x):
        """The same as __call__ without the overhead"""
        return self._legendre_polynomial(x)

    def get_domain(self, Ein=None):
        return -1., 1.

    def integrate(self, lo, hi):
        # Quadrature integration will be exact, so lets just do that.
        xlo = max(-1., lo)
        xhi = min(1., hi)

        # Use the legendre order to find the value of n for the quadrature
        quad_order = int(ceil(0.5 * (float(self.coefficients.shape[0]) + 1.)))

        # Just use Scipy's integration methods until this is identified as a
        # bottleneck
        result = sint.fixed_quad(self, xlo, xhi, n=quad_order)

        return result

    def to_xml_element(self, element_name):
        raise NotImplementedError


class Mixture(Univariate):
    """Probability distribution characterized by a mixture of random variables.

    Parameters
    ----------
    probability : Iterable of Real
        Probability of selecting a particular distribution
    distribution : Iterable of Univariate
        List of distributions with corresponding probabilities

    Attributes
    ----------
    probability : Iterable of Real
        Probability of selecting a particular distribution
    distribution : Iterable of Univariate
        List of distributions with corresponding probabilities

    """

    def __init__(self, probability, distribution):
        super(Mixture, self).__init__()
        self.probability = probability
        self.distribution = distribution

    def __call__(self, x):
        value = 0.
        for i in range(len(self.probability)):
            value += self.probability[i] * self.distribution[i](x)
        return value

    def __len__(self):
        return sum(len(d) for d in self.distribution)

    @property
    def probability(self):
        return self._probability

    @property
    def distribution(self):
        return self._distribution

    @probability.setter
    def probability(self, probability):
        cv.check_type('mixture distribution probabilities', probability,
                      Iterable, Real)
        for p in probability:
            cv.check_greater_than('mixture distribution probabilities',
                                  p, 0.0, True)
        self._probability = probability

    @distribution.setter
    def distribution(self, distribution):
        cv.check_type('mixture distribution components', distribution,
                      Iterable, Univariate)
        self._distribution = distribution

    def get_domain(self, Ein=None):
        low, high = (np.inf, -np.inf)
        for distrib in self.distribution:
            my_low, my_high = distrib.domain
            low = min(low, my_low)
            high = max(high, my_high)
        return (low, high)

    def integrate(self, lo, hi):
        value = 0.
        for i in range(len(self.probability)):
            value += self.probability[i] * \
                self.distribution[i].integrate(lo, hi)
        return value

    def to_xml_element(self, element_name):
        raise NotImplementedError
