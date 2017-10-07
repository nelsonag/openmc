from numbers import Integral, Real
import os

import numpy as np
import h5py

from openmc.data import K_BOLTZMANN, IncidentNeutron, ThermalScattering, \
    DataLibrary
from openmc.mgxs import EnergyGroups
import openmc.checkvalue as cv
from . import NDPP_VERSION, NDPP_FILETYPE
from .ndpp import Ndpp, _SCATTER_REPRESENTATION, _FREEGAS_METHODS


class NdppLibrary(object):
    """Pre-processed nuclear data file used for improved tallying efficiency
    in an OpenMC simulation.

    Parameters
    ----------
    names : List of str
        Library names to process for this NDPP Library; the names must
        correspond to those in the provided cross_sections file
    cross_sections_xml : str
        Path to the cross_sections.xml file
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
    minimum_relative_threshold : float, optional
        The minimum threshold for which outgoing data will be included in the
        output files. Defaults to 1.E-6.

    Attributes
    ----------
    names : List of str
        The name of all the data libraries contained within this NDPP Library
    filepaths : List of str
        Paths to each dataset's hdf5 file
    cross_sections_xml : str
        Path to the cross_sections.xml file
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
    minimum_relative_threshold : float
        The minimum threshold for which outgoing data will be included in the
        output files.
    ndpps: Iterable of openmc.Ndpp
        Pre-processed data relating to the datasets within `libraries`.

    """

    def __init__(self, names, cross_sections_xml, group_structure,
                 scatter_format, order, kTs=None, num_threads=None,
                 tolerance=0.001, freegas_cutoff=None, freegas_method='cxs',
                 minimum_relative_threshold=1.e-6):
        self.cross_sections_xml = cross_sections_xml
        library = DataLibrary.from_xml(cross_sections_xml)
        filepaths = []
        are_neutron = []

        for name in names:
            item = library.get_by_material(name)
            if item is not None:
                # Then we found it
                filepaths.append(item['path'])
                if item['type'] == 'neutron':
                    are_neutron.append(True)
                else:
                    are_neutron.append(False)
            else:
                raise ValueError(name + " is not present in " +
                                 cross_sections_xml)

        self.names = names
        self.filepaths = filepaths
        self.are_neutron = are_neutron
        self.group_structure = group_structure
        self.scatter_format = scatter_format
        self.order = order
        self.kTs = kTs
        self.num_threads = num_threads
        self.tolerance = tolerance
        self.freegas_cutoff = freegas_cutoff
        self.freegas_method = freegas_method
        self.minimum_relative_threshold = minimum_relative_threshold

        self.ndpps = []
        for name, filepath, is_neutron in zip(self.names, self.filepaths,
                                              self.are_neutron):
            self.ndpps.append(Ndpp(name, filepath, is_neutron,
                                   self.group_structure, self.scatter_format,
                                   self.order, self.kTs, self.num_threads,
                                   self.tolerance, self.freegas_cutoff,
                                   self.freegas_method))

    @property
    def cross_sections_xml(self):
        return self._cross_sections_xml

    @property
    def names(self):
        return self._names

    @property
    def filepaths(self):
        return self._filepaths

    @property
    def are_neutron(self):
        return self._are_neutron

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
    def minimum_relative_threshold(self):
        return self._minimum_relative_threshold

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

    @cross_sections_xml.setter
    def cross_sections_xml(self, cross_sections_xml):
        cv.check_type('cross_sections_xml', cross_sections_xml, str)
        self._cross_sections_xml = cross_sections_xml

    @names.setter
    def names(self, names):
        cv.check_iterable_type('names', names, str)
        self._names = names

    @filepaths.setter
    def filepaths(self, filepaths):
        cv.check_iterable_type('filepaths', filepaths, str)
        self._filepaths = filepaths

    @are_neutron.setter
    def are_neutron(self, are_neutron):
        cv.check_iterable_type('are_neutron', are_neutron, bool)
        self._are_neutron = are_neutron

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

    @minimum_relative_threshold.setter
    def minimum_relative_threshold(self, minimum_relative_threshold):
        cv.check_type('minimum_relative_threshold',
                      minimum_relative_threshold, Real)
        cv.check_greater_than('minimum_relative_threshold',
                              minimum_relative_threshold, 0., equality=True)
        self._minimum_relative_threshold = minimum_relative_threshold

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
                          keep_in_memory=True):
        """Create an hdf5 file that can be used for a simulation by first
        computing and then writing the isotopic data.

        Parameters
        ----------
        path : str
            Path to write HDF5 file to; defaults to 'ndpp_lib.h5'
        mode : {'w', 'w-', 'x', 'a'}
            Mode that is used to open the HDF5 file.
            This is the second argument to the :class:`h5py.File` constructor.
        keep_in_memory : logical
            Whether or not to keep each Ndpp object in RAM after it has been
            written to disk. Defaults to False to reduce RAM consumption

        """

        # Write the meta-data
        f = _open_for_writing(self, path, mode)

        for n, ndpp in enumerate(self.ndpps):
            if mode == 'a' and ndpp.name in f:
                print("Bypassing " + ndpp.name)
            else:
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

    def create_macroscopic_library(self, model,
                                   material_path='macro_ndpp_lib.h5',
                                   mode='a'):
        """Create an hdf5 file with NDPP objects specific to the material data
        itself.

        Parameters
        ----------
        model : openmc.Model
            Material information used to create the materials
        path : str
            Path to write HDF5 file to; defaults to 'macro_ndpp_lib.h5'
        mode : {'r', r+', 'w', 'x', 'a'}
            Mode that is used to open the HDF5 file.
            This is the second argument to the :class:`h5py.File` constructor.
        """

        pass

        # f = _open_for_writing(self, material_path, mode)

        # # Get default temperature
        # if model.settings.temperature:
        #     default_kT = model.settings.temperature['default']
        # else:
        #     default_kT = 293.6
        # default_strT = "{}K".format(int(round(default_kT / K_BOLTZMANN)))

        # # Go through the geometry and get a list of all used materials and
        # # their temperatures
        # materials = []
        # cells = model.geometry.get_all_cells()
        # for cell in cells.values():
        #     cell_materials = cell.get_all_materials()
        #     for cell_material in cell_materials:
        #         if cell_material not in materials:
        #         else:
        #             if isinstance(cell.temperature, Iterable):
        #                 materials[cell_material]['kTs'].update(cell.)

        # materials = model.materials

        # for material in materials:
        #     nuclides = material.get_nuclide_atom_densities()

        #     # Get the combined energy grids from the microscopic library
        #     inelastic_energy = None
        #     elastic_energy = None
        #     chi_energy = None
        #     for nuclide in nuclides:
        #         nuclide_ndpp = self.get_by_name(nuclide)

        #     for nuclide in nuclides:
        #         atom_density = nuclide[1]

        #         micro_ndpp = self.get_by_name(nuclide)

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

        # Write the meta-data
        f = _open_for_writing(self, path, mode)

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

        # Get cross section library
        cross_sections_xml = f.attrs['cross_sections_xml'].decode()

        # Get names
        names = []
        for name in f.items():
            names.append(name)

        group_edges = f['group structure'].value
        group_structure = EnergyGroups(group_edges)
        scatter_format = f.attrs['scatter_format'].decode()
        if 'kTs' in f:
            kTs = f['kTs'].value
        else:
            kTs = None
        num_threads = f.attrs['num_threads']
        tolerance = f.attrs['tolerance']
        minimum_relative_threshold = f.attrs['minimum_relative_threshold']
        order = f.attrs['order']
        freegas_cutoff = f.attrs['freegas_cutoff']
        freegas_method = f.attrs['freegas_method'].decode()

        # Initialize the object
        data = cls(names, cross_sections_xml, group_structure, scatter_format,
                   order, kTs=kTs, num_threads=num_threads,
                   tolerance=tolerance, freegas_cutoff=freegas_cutoff,
                   freegas_method=freegas_method,
                   minimum_relative_threshold=minimum_relative_threshold)

        for ndpp in data.ndpps:
            ndpp.from_hdf5(f[ndpp.name])

        return data


def _open_for_writing(this, path, mode):
    cv.check_type('path', path, str)

    if mode not in ['w', 'w-', 'x', 'a']:
        raise ValueError("Invalid option for mode")

    if os.path.isfile(path):
        exists = True
    else:
        exists = False

    # Open file
    f = h5py.File(path, mode, libver='latest')

    if mode == 'a' and exists:
        # Then we need to make sure all the processing parameters match
        # Check filetype and version
        cv.check_filetype_version(f, NDPP_FILETYPE, NDPP_VERSION[0])

        fail = False

        # Get cross section library
        cross_sections_xml = f.attrs['cross_sections_xml'].decode()
        if cross_sections_xml != this.cross_sections_xml:
            fail = True

        group_edges = f['group structure'].value
        group_structure = EnergyGroups(group_edges)
        if group_structure != this.group_structure:
            fail = True

        scatter_format = f.attrs['scatter_format'].decode()
        if scatter_format != this.scatter_format:
            fail = True

        if 'kTs' in f:
            kTs = f['kTs'].value
        else:
            kTs = None
        if kTs != this.kTs:
            fail = True

        tolerance = f.attrs['tolerance']
        if tolerance != this.tolerance:
            fail = True

        minimum_relative_threshold = f.attrs['minimum_relative_threshold']
        if minimum_relative_threshold != this.minimum_relative_threshold:
            fail = True

        order = f.attrs['order']
        if order != this.order:
            fail = True

        freegas_cutoff = f.attrs['freegas_cutoff']
        if freegas_cutoff != this.freegas_cutoff:
            fail = True

        freegas_method = f.attrs['freegas_method'].decode()
        if freegas_method != this.freegas_method:
            fail = True

        if fail:
            raise ValueError("Cannot append to {} since it was written "
                             "with different parameters".format(path))
    else:
        f.attrs['filetype'] = np.string_('ndpp')
        f.attrs['version'] = np.array(NDPP_VERSION)
        f.attrs['cross_sections_xml'] = np.string_(this.cross_sections_xml)
        f.create_dataset('group structure', data=this.group_edges)
        if this.kTs is not None:
            f.create_dataset('kTs', data=this.kTs)
        f.attrs['num_threads'] = this.num_threads
        f.attrs['tolerance'] = this.tolerance
        f.attrs['minimum_relative_threshold'] = \
            this.minimum_relative_threshold
        f.attrs['scatter_format'] = np.string_(this.scatter_format)
        f.attrs['order'] = this.order
        if this.freegas_cutoff is None:
            f.attrs['freegas_cutoff'] = -1
        else:
            f.attrs['freegas_cutoff'] = this.freegas_cutoff
        f.attrs['freegas_method'] = np.string_(this.freegas_method)

    return f
