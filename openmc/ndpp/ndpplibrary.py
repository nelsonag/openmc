from numbers import Integral, Real

import numpy as np
import h5py

from openmc.data import K_BOLTZMANN, IncidentNeutron, ThermalScattering, \
    DataLibrary
# from openmc.data.neutron import IncidentNeutron
# from openmc.data.thermal import ThermalScattering
from openmc.mgxs import EnergyGroups
import openmc.checkvalue as cv
from . import NDPP_VERSION, NDPP_FILETYPE
from .ndpp import Ndpp, _SCATTER_REPRESENTATION, _FREEGAS_METHODS


class NdppLibrary(object):
    """Pre-processed nuclear data file used for improved tallying efficiency
    in an OpenMC simulation.

    Parameters
    ----------
    libraries : openmc.data.DataLibrary, or Iterable of
                openmc.data.IncidentNeutron and openmc.data.ThermalScattering
                objects
        Data libraries to process for this NDPP Library
    group_structure : openmc.mgxs.EnergyGroups
        Energy group structure for energy condensation
    scatter_format : {'legendre', 'histogram'}
        Mathematical scatter_format of the scattering matrices: either a
        Legendre expansion of the order defined in the order parameter, or
        a histogram scatter_format with a number of bins defined by the order
        parameter.
    order : int
        Either the Legendre expansion order or the number of histogram bins.
    kTs : Iterable of float, optional
        Subset of temperatures (in eV) to process from the continuous-energy
        data.  Defaults to processing all temperatures in the data library
    num_threads : int, optional
        Number of threads to use for the parallel calculation. Defaults to all
        CPU threads available.
    tolerance : float, optional
        Interpolation tolerance to apply to the incoming energy grid
        constructed by this class. Defaults to 0.001, which is equivalent to a
        0.1% error.
    freegas_cutoff : float, optional
        Multiplier of temperature (kT) in which the free-gas kernel is applied;
        defaults to 400 unless the `library` is for H-1, then a large value is
        used so all energies are treated with this method.
    freegas_method : {'cxs' or 'doppler'}, optional
        The method to be used for the cross section of the target. If `cxs` is
        provided, the constant cross-section free-gas kernel will be used. If
        `doppler` is provided, then the cross section variation is included in
        the free-gas kernel. The `doppler` method can only be used if `0K`
        elastic scattering data is present in the `library`. Defaults to `cxs`.

    Attributes
    ----------
    libraries : openmc.data.DataLibrary, or Iterable of
                openmc.data.IncidentNeutron and openmc.data.ThermalScattering
                objects
        Data libraries to process for this NDPP Library
    group_structure : openmc.mgxs.EnergyGroups
        Energy group structure for energy condensation
    scatter_format : {'legendre' or 'histogram'}
        Mathematical scatter_format of the scattering matrices: either a
        Legendre expansion of the order defined in the order parameter, or
        a histogram scatter_format with a number of bins defined by the order
        parameter.
    order : int
        Either the Legendre expansion order or the number of histogram bins.
    kTs : Iterable of float
        Subset of temperatures (in eV) to process from the continuous-energy
        data.
    num_threads : int
        Number of threads to use for the parallel calculation.
    tolerance : float
        Interpolation tolerance to apply to the incoming energy grid
        constructed by this class.
    freegas_cutoff : float
        Multiplier of temperature (kT) in which the free-gas kernel is applied
    freegas_method : {'cxs' or 'doppler'}
        The method to be used for the cross section of the target. If `cxs` is
        provided, the constant cross-section free-gas kernel will be used. If
        `doppler` is provided, then the cross section variation is included in
        the free-gas kernel. The `doppler` method can only be used if `0K`
        elastic scattering data is present in the `library`.
    names : List of str
        The name of all the data libraries contained within this NDPP Library
    ndpps: Iterable of openmc.Ndpp
        Pre-processed data relating to the datasets within `libraries`.

    """

    def __init__(self, libraries, group_structure, scatter_format,
                 order, kTs=None, num_threads=None, tolerance=0.001,
                 freegas_cutoff=None, freegas_method='cxs'):
        if isinstance(libraries, DataLibrary):
            datasets = []
            for i, item in enumerate(libraries.libraries):
                if item['type'] == 'neutron':
                    the_type = IncidentNeutron
                else:
                    the_type = ThermalScattering
                h5file = h5py.File(item['path'], 'r')
                for material in item['materials']:
                    datasets.append(the_type.from_hdf5(h5file[material]))
        else:
            datasets = libraries
        self.libraries = datasets
        self.group_structure = group_structure
        self.scatter_format = scatter_format
        self.order = order
        self.kTs = kTs
        self.num_threads = num_threads
        self.tolerance = tolerance
        self.freegas_cutoff = freegas_cutoff
        self.freegas_method = freegas_method

        self.ndpps = []
        for library in self.libraries:
            self.ndpps.append(Ndpp(library, self.group_structure,
                                   self.scatter_format, self.order, self.kTs,
                                   self.num_threads, self.tolerance,
                                   self.freegas_cutoff, self.freegas_method))

    @property
    def libraries(self):
        return self._libraries

    @property
    def group_structure(self):
        return self._group_structure

    @property
    def scatter_format(self):
        return self._scatter_format

    @property
    def order(self):
        return self._order

    @property
    def kTs(self):
        return self._kTs

    @property
    def num_threads(self):
        return self._num_threads

    @property
    def tolerance(self):
        return self._tolerance

    @property
    def freegas_cutoff(self):
        return self._freegas_cutoff

    @property
    def freegas_method(self):
        return self._freegas_method

    @property
    def num_groups(self):
        return self.group_structure.num_groups

    @property
    def group_edges(self):
        return self.group_structure.group_edges

    @property
    def num_angle(self):
        """Number of angular bins/expansion orders to use
        """
        if self.scatter_format == 'legendre':
            return self.order + 1
        else:
            return self.order

    @property
    def temperatures(self):
        if self.kTs is None:
            return None
        return ["{}K".format(int(round(kT / K_BOLTZMANN))) for kT in self.kTs]

    @libraries.setter
    def libraries(self, libraries):
        cv.check_iterable_type('libraries', libraries,
                               (IncidentNeutron, ThermalScattering))
        self._libraries = libraries

    @group_structure.setter
    def group_structure(self, group_structure):
        cv.check_type('group_structure', group_structure, EnergyGroups)
        self._group_structure = group_structure

    @scatter_format.setter
    def scatter_format(self, scatter_format):
        cv.check_value('scatter_format', scatter_format,
                       _SCATTER_REPRESENTATION)
        self._scatter_format = scatter_format

    @order.setter
    def order(self, order):
        cv.check_type('order', order, Integral)
        if self.scatter_format is 'legendre':
            cv.check_greater_than('order', order, 0, equality=True)
            cv.check_less_than('order', order, 10, equality=True)
        elif self.scatter_format is 'histogram':
            cv.check_greater_than('order', order, 0, equality=False)
        self._order = order

    @kTs.setter
    def kTs(self, kTs):
        if kTs is not None:
            cv.check_iterable_type('kTs', kTs, Real)
            # Make sure kTs are in the temperature set of the library
            for kT in kTs:
                cv.check_value('kT in kTs', kT, self.library.kTs)
        self._kTs = kTs

    @num_threads.setter
    def num_threads(self, num_threads):
        if num_threads is not None:
            cv.check_type('num_threads', num_threads, Integral)
            cv.check_greater_than('num_threads', num_threads, 0, equality=True)
        self._num_threads = num_threads

    @tolerance.setter
    def tolerance(self, tolerance):
        cv.check_type('tolerance', tolerance, Real)
        cv.check_greater_than('tolerance', tolerance, 0.,
                              equality=True)
        self._tolerance = tolerance

    @freegas_cutoff.setter
    def freegas_cutoff(self, freegas_cutoff):
        if freegas_cutoff is not None:
            cv.check_type('freegas_cutoff', freegas_cutoff, Real)
            cv.check_greater_than('freegas_cutoff', freegas_cutoff, 0.,
                                  equality=True)
        self._freegas_cutoff = freegas_cutoff

    @freegas_method.setter
    def freegas_method(self, freegas_method):
        cv.check_value('freegas_method', freegas_method,
                       _FREEGAS_METHODS)
        self._freegas_method = freegas_method

    @property
    def names(self):
        return [ndpp.name for ndpp in self.ndpps]

    def get_by_name(self, name):
        """Access the NDPP objects by name

        Parameters
        ----------
        name : str
            Name of openmc.Ndpp object to obtain

        Returns
        -------
        result : openmc.Ndpp or None
            Provides the matching NDPP object or None, if not found

        """
        cv.check_type("name", name, str)
        result = None
        for ndpp in self.ndpps:
            if name == ndpp.name:
                result = ndpp
        return result

    def process_and_write(self, path='ndpp_lib.h5', mode='a',
                          keep_in_memory=False):
        """Create an hdf5 file that can be used for a simulation by first
        computing and then writing the isotopic data.

        Parameters
        ----------
        path : str
            Path to write HDF5 file to; defaults to 'ndpp_lib.h5'
        mode : {'r', r+', 'w', 'x', 'a'}
            Mode that is used to open the HDF5 file.
            This is the second argument to the :class:`h5py.File` constructor.
        keep_in_memory : logical
            Whether or not to keep each Ndpp object in RAM after it has been
            written to disk. Defaults to False to reduce RAM consumption

        """

        cv.check_type('path', path, str)

        # Create and write to the HDF5 file
        # Open file and write version
        f = h5py.File(path, mode, libver='latest')
        f.attrs['filetype'] = np.string_('ndpp')
        f.attrs['version'] = np.array(NDPP_VERSION)
        f.create_dataset('group structure', data=self.group_edges)
        f.attrs['scatter_format'] = np.string_(self.scatter_format)
        f.attrs['order'] = self.order
        if self.freegas_cutoff is None:
            f.attrs['freegas_cutoff'] = -1
        else:
            f.attrs['freegas_cutoff'] = self.freegas_cutoff
        f.attrs['freegas_method'] = np.string_(self.freegas_method)

        for n, ndpp in enumerate(self.ndpps):
            print("Evaluating " + ndpp.name)
            ndpp.process()
            ndpp.to_hdf5(f)
            if not keep_in_memory:
                self.ndpps[n] = None

        f.close()

    def process(self):
        for ndpp in self.ndpps:
            print("Evaluating " + ndpp.name)
            ndpp.process()

    def export_to_hdf5(self, path='ndpp_lib.h5', mode='a'):
        """Create an hdf5 file that can be used for a simulation.

        Parameters
        ----------
        path : str
            Path to write HDF5 file to; defaults to 'ndpp_lib.h5'
        mode : {'r', r+', 'w', 'x', 'a'}
            Mode that is used to open the HDF5 file.
            This is the second argument to the :class:`h5py.File` constructor.

        """

        cv.check_type('path', path, str)

        # Create and write to the HDF5 file
        # Open file and write version
        f = h5py.File(path, mode, libver='latest')
        f.attrs['filetype'] = np.string_('ndpp')
        f.attrs['version'] = np.array(NDPP_VERSION)
        f.create_dataset('group structure', data=self.group_edges)
        f.attrs['scatter_format'] = np.string_(self.scatter_format)
        f.attrs['order'] = self.order
        f.attrs['freegas_cutoff'] = self.freegas_cutoff
        f.attrs['freegas_method'] = np.string_(self.freegas_method)

        for ndpp in self.ndpps:
            ndpp.to_hdf5(f)

        f.close()

    @classmethod
    def from_hdf5(cls, filename=None):
        """Generate an NDPP Library from an HDF5 group or file

        Parameters
        ----------
        filename : str
            Name of HDF5 file containing NDPP data

        Returns
        -------
        openmc.NDPPLibrary
            Pre-processed nuclear data object.

        """

        cv.check_type('filename', filename, str)
        f = h5py.File(filename, 'r')

        # Check filetype and version
        cv.check_filetype_version(f, NDPP_FILETYPE, NDPP_VERSION)

        group_edges = f.attrs['group structure']
        group_structure = openmc.mgxs.EnergyGroups(group_edges)
        scatter_format = f.attrs['scatter_format'].decode()
        order = f.attrs['order']
        freegas_cutoff = f.attrs['freegas_cutoff']
        freegas_method = f.attrs['freegas_method'].decode()

        # Initialize the object
        data = cls(libraries, group_structure, scatter_format, order)
        # (self, libraries, group_structure, scatter_format,
        #          order, kTs=None, num_threads=None, tolerance=0.001,
        #          freegas_cutoff=None, freegas_method='cxs')

        for group_name, group in f.items():
            data.add_xsdata(openmc.XSdata.from_hdf5(group, group_name,
                                                    energy_groups,
                                                    num_delayed_groups))

        return data
