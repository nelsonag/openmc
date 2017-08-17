from abc import ABCMeta, abstractmethod
from collections import Iterable, Callable
from numbers import Real, Integral

from six import add_metaclass
import numpy as np

import openmc.data
import openmc.stats
import openmc.checkvalue as cv
from openmc.mixin import EqualityMixin
from .data import EV_PER_MEV
try:
    from .function_methods_cython import *
except ImportError:
    from .function_methods import *

INTERPOLATION_SCHEME = {1: 'histogram', 2: 'linear-linear', 3: 'linear-log',
                        4: 'log-linear', 5: 'log-log'}


@add_metaclass(ABCMeta)
class Function1D(EqualityMixin):
    """A function of one independent variable with HDF5 support."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def to_hdf5(self, group, name='xy'):
        """Write function to an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to
        name : str
            Name of the dataset to create

        """
        pass

    @classmethod
    def from_hdf5(cls, dataset):
        """Generate function from an HDF5 dataset

        Parameters
        ----------
        dataset : h5py.Dataset
            Dataset to read from

        Returns
        -------
        openmc.data.Function1D
            Function read from dataset

        """
        for subclass in cls.__subclasses__():
            if dataset.attrs['type'].decode() == subclass.__name__:
                return subclass.from_hdf5(dataset)
        raise ValueError("Unrecognized Function1D class: '" +
                         dataset.attrs['type'].decode() + "'")


class Tabulated1D(Function1D):
    """A one-dimensional tabulated function.

    This class mirrors the TAB1 type from the ENDF-6 format. A tabulated
    function is specified by tabulated (x,y) pairs along with interpolation
    rules that determine the values between tabulated pairs.

    Once an object has been created, it can be used as though it were an actual
    function, e.g.:

    >>> f = Tabulated1D([0, 10], [4, 5])
    >>> [f(xi) for xi in numpy.linspace(0, 10, 5)]
    [4.0, 4.25, 4.5, 4.75, 5.0]

    Parameters
    ----------
    x : Iterable of float
        Independent variable
    y : Iterable of float
        Dependent variable
    breakpoints : Iterable of int
        Breakpoints for interpolation regions
    interpolation : Iterable of int
        Interpolation scheme identification number, e.g., 3 means y is linear in
        ln(x).

    Attributes
    ----------
    x : Iterable of float
        Independent variable
    y : Iterable of float
        Dependent variable
    breakpoints : Iterable of int
        Breakpoints for interpolation regions
    interpolation : Iterable of int
        Interpolation scheme identification number, e.g., 3 means y is linear in
        ln(x).
    n_regions : int
        Number of interpolation regions
    n_pairs : int
        Number of tabulated (x,y) pairs

    """

    def __init__(self, x, y, breakpoints=None, interpolation=None):
        if breakpoints is None or interpolation is None:
            # Single linear-linear interpolation region by default
            self.breakpoints = np.array([len(x)])
            self.interpolation = np.array([2])
        else:
            self.breakpoints = np.asarray(breakpoints, dtype=int)
            self.interpolation = np.asarray(interpolation, dtype=int)

        self.x = np.asarray(x)
        self.y = np.asarray(y)

    def __call__(self, x):
        return tabulated1d_call(self, x)

    def __len__(self):
        return len(self.x)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def breakpoints(self):
        return self._breakpoints

    @property
    def interpolation(self):
        return self._interpolation

    @property
    def n_pairs(self):
        return len(self.x)

    @property
    def n_regions(self):
        return len(self.breakpoints)

    @x.setter
    def x(self, x):
        cv.check_type('x values', x, Iterable, Real)
        self._x = x

    @y.setter
    def y(self, y):
        cv.check_type('y values', y, Iterable, Real)
        self._y = y

    @breakpoints.setter
    def breakpoints(self, breakpoints):
        cv.check_type('breakpoints', breakpoints, Iterable, Integral)
        self._breakpoints = breakpoints

    @interpolation.setter
    def interpolation(self, interpolation):
        cv.check_type('interpolation', interpolation, Iterable, Integral)
        self._interpolation = interpolation

    def get_domain(self, Ein=None):
        return (self._x[0], self._x[-1])

    def integral(self):
        """Integral of the tabulated function over its tabulated range.

        Returns
        -------
        numpy.ndarray
            Array of same length as the tabulated data that represents partial
            integrals from the bottom of the range to each tabulated point.

        """

        # Create output array
        partial_sum = np.zeros(len(this.x) - 1)

        i_low = 0
        for k in range(len(this.breakpoints)):
            # Determine which x values are within this interpolation range
            i_high = this.breakpoints[k] - 1

            # Get x values and bounding (x,y) pairs
            x0 = this.x[i_low:i_high]
            x1 = this.x[i_low + 1:i_high + 1]
            y0 = this.y[i_low:i_high]
            y1 = this.y[i_low + 1:i_high + 1]

            if this.interpolation[k] == 1:
                # Histogram
                partial_sum[i_low:i_high] = y0 * (x1 - x0)

            elif this.interpolation[k] == 2:
                # Linear-linear
                m = (y1 - y0) / (x1 - x0)
                partial_sum[i_low:i_high] = (y0 - m * x0) * (x1 - x0) + \
                    m * (x1**2 - x0**2) / 2

            elif this.interpolation[k] == 3:
                # Linear-log
                logx = np.log(x1 / x0)
                m = (y1 - y0) / logx
                partial_sum[i_low:i_high] = y0 + m * (x1 * (logx - 1) + x0)

            elif this.interpolation[k] == 4:
                # Log-linear
                m = np.log(y1 / y0) / (x1 - x0)
                partial_sum[i_low:i_high] = y0 / m * (np.exp(m * (x1 - x0)) -
                                                      1)

            elif this.interpolation[k] == 5:
                # Log-log
                m = np.log(y1 / y0) / np.log(x1 / x0)
                partial_sum[i_low:i_high] = y0 / ((m + 1) * x0**m) * (
                    x1**(m + 1) - x0**(m + 1))

            i_low = i_high

        return np.concatenate(([0.], np.cumsum(partial_sum)))

    def to_hdf5(self, group, name='xy'):
        """Write tabulated function to an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to
        name : str
            Name of the dataset to create

        """
        dataset = group.create_dataset(name, data=np.vstack(
            [self.x, self.y]))
        dataset.attrs['type'] = np.string_(type(self).__name__)
        dataset.attrs['breakpoints'] = self.breakpoints
        dataset.attrs['interpolation'] = self.interpolation

    @classmethod
    def from_hdf5(cls, dataset):
        """Generate tabulated function from an HDF5 dataset

        Parameters
        ----------
        dataset : h5py.Dataset
            Dataset to read from

        Returns
        -------
        openmc.data.Tabulated1D
            Function read from dataset

        """
        if dataset.attrs['type'].decode() != cls.__name__:
            raise ValueError("Expected an HDF5 attribute 'type' equal to '" +
                             cls.__name__ + "'")

        x = dataset.value[0, :]
        y = dataset.value[1, :]
        breakpoints = dataset.attrs['breakpoints']
        interpolation = dataset.attrs['interpolation']
        return cls(x, y, breakpoints, interpolation)

    @classmethod
    def from_ace(cls, ace, idx=0, convert_units=True):
        """Create a Tabulated1D object from an ACE table.

        Parameters
        ----------
        ace : openmc.data.ace.Table
            An ACE table
        idx : int
            Offset to read from in XSS array (default of zero)
        convert_units : bool
            If the abscissa represents energy, indicate whether to convert MeV
            to eV.

        Returns
        -------
        openmc.data.Tabulated1D
            Tabulated data object

        """

        # Get number of regions and pairs
        n_regions = int(ace.xss[idx])
        n_pairs = int(ace.xss[idx + 1 + 2*n_regions])

        # Get interpolation information
        idx += 1
        if n_regions > 0:
            breakpoints = ace.xss[idx:idx + n_regions].astype(int)
            interpolation = ace.xss[idx + n_regions:idx + 2*n_regions].astype(int)
        else:
            # 0 regions implies linear-linear interpolation by default
            breakpoints = np.array([n_pairs])
            interpolation = np.array([2])

        # Get (x,y) pairs
        idx += 2*n_regions + 1
        x = ace.xss[idx:idx + n_pairs].copy()
        y = ace.xss[idx + n_pairs:idx + 2*n_pairs].copy()

        if convert_units:
            x *= EV_PER_MEV

        return Tabulated1D(x, y, breakpoints, interpolation)


class Polynomial(np.polynomial.Polynomial, Function1D):
    def get_domain(self, Ein=None):
        return self.domain

    def to_hdf5(self, group, name='xy'):
        """Write polynomial function to an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to
        name : str
            Name of the dataset to create

        """
        dataset = group.create_dataset(name, data=self.coef)
        dataset.attrs['type'] = np.string_(type(self).__name__)

    @classmethod
    def from_hdf5(cls, dataset):
        """Generate function from an HDF5 dataset

        Parameters
        ----------
        dataset : h5py.Dataset
            Dataset to read from

        Returns
        -------
        openmc.data.Function1D
            Function read from dataset

        """
        if dataset.attrs['type'].decode() != cls.__name__:
            raise ValueError("Expected an HDF5 attribute 'type' equal to '"
                             + cls.__name__ + "'")
        return cls(dataset.value)


class Combination(EqualityMixin):
    """Combination of multiple functions with a user-defined operator

    This class allows you to create a callable object which represents the
    combination of other callable objects by way of a series of user-defined
    operators connecting each of the callable objects.

    Parameters
    ----------
    functions : Iterable of Callable
        Functions to combine according to operations
    operations : Iterable of numpy.ufunc
        Operations to perform between functions; note that the standard order
        of operations will not be followed, but can be simulated by
        combinations of Combination objects. The operations parameter must have
        a length one less than the number of functions.


    Attributes
    ----------
    functions : Iterable of Callable
        Functions to combine according to operations
    operations : Iterable of numpy.ufunc
        Operations to perform between functions; note that the standard order
        of operations will not be followed, but can be simulated by
        combinations of Combination objects. The operations parameter must have
        a length one less than the number of functions.

    """

    def __init__(self, functions, operations):
        self.functions = functions
        self.operations = operations

    def __call__(self, x):
        ans = self.functions[0](x)
        for i, operation in enumerate(self.operations):
            ans = operation(ans, self.functions[i + 1](x))
        return ans

    @property
    def functions(self):
        return self._functions

    @functions.setter
    def functions(self, functions):
        cv.check_type('functions', functions, Iterable, Callable)
        self._functions = functions

    @property
    def operations(self):
        return self._operations

    @operations.setter
    def operations(self, operations):
        cv.check_type('operations', operations, Iterable, np.ufunc)
        length = len(self.functions) - 1
        cv.check_length('operations', operations, length, length_max=length)
        self._operations = operations

    def get_domain(self, Ein=None):
        low, high = (np.inf, -np.inf)
        for func in self.functions:
            my_low, my_high = func.domain
            low = min(low, my_low)
            high = max(high, my_high)
        return (low, high)


class Sum(EqualityMixin):
    """Sum of multiple functions.

    This class allows you to create a callable object which represents the sum
    of other callable objects. This is used for summed reactions whereby the
    cross section is defined as the sum of other cross sections.

    Parameters
    ----------
    functions : Iterable of Callable
        Functions which are to be added together

    Attributes
    ----------
    functions : Iterable of Callable
        Functions which are to be added together

    """

    def __init__(self, functions):
        self.functions = functions

    def __call__(self, x):
        return sum(f(x) for f in self.functions)

    @property
    def functions(self):
        return self._functions

    @functions.setter
    def functions(self, functions):
        cv.check_type('functions', functions, Iterable, Callable)
        self._functions = functions

    def get_domain(self, Ein=None):
        low, high = (np.inf, -np.inf)
        for func in self.functions:
            my_low, my_high = func.domain
            low = min(low, my_low)
            high = max(high, my_high)
        return (low, high)


class Regions1D(EqualityMixin):
    """Piecewise composition of multiple functions.

    This class allows you to create a callable object which is composed
    of multiple other callable objects, each applying to a specific interval

    Parameters
    ----------
    functions : Iterable of Callable
        Functions which are to be combined in a piecewise fashion
    breakpoints : Iterable of float
        The values of the dependent variable that define the domain of
        each function. The *i*th and *(i+1)*th values are the limits of the
        domain of the *i*th function. Values must be monotonically increasing.

    Attributes
    ----------
    functions : Iterable of Callable
        Functions which are to be combined in a piecewise fashion
    breakpoints : Iterable of float
        The breakpoints between each function

    """

    def __init__(self, functions, breakpoints):
        self.functions = functions
        self.breakpoints = breakpoints

    def __call__(self, x):
        i = np.searchsorted(self.breakpoints, x)
        if isinstance(x, Iterable):
            ans = np.empty_like(x)
            for j in range(len(i)):
                ans[j] = self.functions[i[j]](x[j])
            return ans
        else:
            return self.functions[i](x)

    @property
    def functions(self):
        return self._functions

    @property
    def breakpoints(self):
        return self._breakpoints

    @functions.setter
    def functions(self, functions):
        cv.check_type('functions', functions, Iterable, Callable)
        self._functions = functions

    @breakpoints.setter
    def breakpoints(self, breakpoints):
        cv.check_iterable_type('breakpoints', breakpoints, Real)
        self._breakpoints = breakpoints

    def get_domain(self, Ein=None):
        low_domain = self.functions[0].domain
        high_domain = self.functions[-1].domain
        return (low_domain[0], high_domain[1])


class ResonancesWithBackground(EqualityMixin):
    """Cross section in resolved resonance region.

    Parameters
    ----------
    resonances : openmc.data.Resonances
        Resolved resonance parameter data
    background : Callable
        Background cross section as a function of energy
    mt : int
        MT value of the reaction

    Attributes
    ----------
    resonances : openmc.data.Resonances
        Resolved resonance parameter data
    background : Callable
        Background cross section as a function of energy
    mt : int
        MT value of the reaction

    """


    def __init__(self, resonances, background, mt):
        self.resonances = resonances
        self.background = background
        self.mt = mt

    def __call__(self, x):
        # Get background cross section
        xs = self.background(x)

        for r in self.resonances:
            if not isinstance(r, openmc.data.resonance._RESOLVED):
                continue

            if isinstance(x, Iterable):
                # Determine which energies are within resolved resonance range
                within = (r.energy_min <= x) & (x <= r.energy_max)

                # Get resonance cross sections and add to background
                resonant_xs = r.reconstruct(x[within])
                xs[within] += resonant_xs[self.mt]
            else:
                if r.energy_min <= x <= r.energy_max:
                    resonant_xs = r.reconstruct(x)
                    xs += resonant_xs[self.mt]

        return xs

    @property
    def background(self):
        return self._background

    @property
    def mt(self):
        return self._mt

    @property
    def resonances(self):
        return self._resonances

    @background.setter
    def background(self, background):
        cv.check_type('background cross section', background, Callable)
        self._background = background

    @mt.setter
    def mt(self, mt):
        cv.check_type('MT value', mt, Integral)
        self._mt = mt

    @resonances.setter
    def resonances(self, resonances):
        cv.check_type('resolved resonance parameters', resonances,
                      openmc.data.Resonances)
        self._resonances = resonances
