from __future__ import division, unicode_literals
from collections import MutableMapping
from multiprocessing import Pool
import sys
from numbers import Integral, Real
from functools import reduce

import numpy as np
import scipy.sparse as sps
import h5py

from openmc.data.data import K_BOLTZMANN
from openmc.data.endf import SUM_RULES
from openmc.data.neutron import IncidentNeutron
from openmc.data.thermal import ThermalScattering
from openmc.mgxs import EnergyGroups
import openmc.checkvalue as cv
from . import NDPP_VERSION_MAJOR
from .evaluators import *
from .cython_integrators import set_freegas_method

if sys.version_info[0] >= 3:
    basestring = str

# Options for output scatter_formats for scattering angle distributions
_SCATTER_REPRESENTATION = ['legendre', 'histogram']

# Options for methods to use when calculating the free-gas scattering kernel
_FREEGAS_METHODS = ['cxs', 'doppler']

# Fission reaction channel number
_FISSION_MTS = set([18] + [mt for mt in SUM_RULES[18]])

# Inelastic scattering reaction channel numbers
_INELASTIC_MTS = set([mt for mt in SUM_RULES[3] if mt != 27] +
                     [mt for mt in SUM_RULES[4]])

# Elastic scattering reaction channel number
_ELASTIC_MT = 2

# Number of incoming energy domain subdivisions for parallelizations
_NUM_ENERGY_SUBDIV = 32

# Low energy point to use to avoid division by zero (if 0 energy was used)
_LOW_ENERGY = 1.e-5

# Number of Lagrange interpolation points for kernel quadrature integration
_K_LAGRANGE = 20


class Ndpp(object):
    """Nuclear data pre-processor.  This class takes an OpenMC HDF5 nuclear data
    library and pre-computes the data needed to most efficiently tally
    multi-group cross section data with secondary particle distributions like
    scattering moment matrices and fission energy spectra.

    Parameters
    ----------
    library : openmc.data.IncidentNeutron or openmc.data.ThermalScattering
        Nuclear data to operate on
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
        Subset of temperatures of the target nuclide to process from the
        `library`. The temperatures have units of eV. Default is to process all
        temperatures in the library except for the 0K data
    num_threads : int, optional
        Number of threads to use for the parallel calculation. Defaults to all
        CPU threads available.
    tolerance : float, optional
        Interpolation tolerance to apply to the incoming energy grid
        constructed by this class. Defaults to 0.001, which is equivalent to a
        0.1% error.
    freegas_cutoff : float, optional
        Multiplier of temperature in which the free-gas kernel is applied;
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
        output files. Defaults to 1.E-10.

    Attributes
    ----------
    name : str
        Name of the library in this dataset.
    group_structure : openmc.mgxs.EnergyGroups
        Energy group structure for energy condensation
    library : openmc.data.IncidentNeutron or openmc.data.ThermalScattering
        Nuclear data to operate on
    scatter_format : {'legendre', or 'histogram'}
        Mathematical scatter_format of the scattering matrices: either a
        Legendre expansion of the order defined in the order attribute, or
        a histogram scatter_format with a number of bins defined by the order
        attribute.
    order : int
        Either the Legendre expansion order or the number of histogram bins.
    kTs : Iterable of float
        Subset of temperatures of the target nuclide to process from the
        `library`. The temperatures have units of eV.
    num_threads : int, optional
        Number of threads to use for the parallel calculation.
    tolerance : float
        Interpolation tolerance to apply to the incoming energy grid
        constructed by this class.
    num_groups : int
        Number of energy groups
    elastic : dict of str to np.ndarray
        Elastic scattering data calculated by the pre-processor for later use
        in tallying of multi-group scattering matrices by OpenMC.
        The array has a shape of (len(elastic_energy), num_groups, order). This
        data is provided in a dictionary where the key is the temperature of
        the data set.
    elastic_energy : dict of str to np.ndarray
        Incoming energy grid for the elastic scattering data (in eV). This
        data is provided in a dictionary where the key is the temperature of
        the data set.
    nu_inelastic : np.ndarray
        Inelastic scattering data, inluding the scattering multiplicity
        [(n,xn)], calculated by the pre-processor for later use in tallying of
        multi-group scattering matrices by OpenMC. The array has a shape of
        (len(scatter_energy), num_groups, order)
    inelastic : np.ndarray
        Inelastic scattering data calculated by the pre-processor for later use
        in tallying of multi-group scattering matrices by OpenMC.
        The array has a shape of (len(scatter_energy), num_groups, order)
    inelastic_energy : np.ndarray
        Incoming energy grid for the inelastic scattering data (in eV)
    total_chi : np.ndarray
        Data calculated by the pre-processor for later use in tallying of
        multi-group fission spectra data by OpenMC.
        The array has a shape of (len(chi_energy), num_groups)
    prompt_chi : np.ndarray
        Data calculated by the pre-processor for later use in tallying of
        multi-group prompt fission spectra data by OpenMC.
        The array has a shape of (len(chi_energy), num_groups)
    delayed_chi : np.ndarray
        Data calculated by the pre-processor for later use in tallying of
        multi-group delayed fission spectra data by OpenMC.
        The array has a shape of (len(chi_energy), num_groups)
    chi_energy : np.ndarray
        Incoming energy grid for the fission spectra data (in eV)
    freegas_cutoff : float
        Multiplier of temperature in which the free-gas kernel is applied.
    freegas_method : {'cxs' or 'doppler'}
        The method to be used for the cross section of the target. If `cxs` is
        provided, the constant cross-section free-gas kernel will be used. If
        `doppler` is provided, then the cross section variation is included in
        the free-gas kernel. The `doppler` method can only be used if `0K`
        elastic scattering data is present in the `library`.
    minimum_relative_threshold : float
        The minimum threshold for which outgoing data will be included in the
        output files.

    """

    def __init__(self, library, group_structure, scatter_format,
                 order, kTs=None, num_threads=None, tolerance=0.001,
                 freegas_cutoff=None, freegas_method='cxs',
                 minimum_relative_threshold=1.e-10):
        self.library = library
        self.group_structure = group_structure
        self.scatter_format = scatter_format
        self.order = order
        if kTs is None:
            kTs = [kT for kT in library.kTs if kT > 0.]
        self.kTs = kTs
        self.num_threads = num_threads
        self.tolerance = tolerance
        self.elastic = {}
        self.elastic_energy = {}
        self.nu_inelastic = None
        self.inelastic = None
        self.inelastic_energy = None
        self.total_chi = {}
        self.prompt_chi = {}
        self.delayed_chi = {}
        self.chi_energy = {}
        self.fissionable = False
        self.num_delayed_groups = 0

        if freegas_cutoff is None:
            if library.atomic_weight_ratio < 1.:
                # Set this to a number large enough so all Ein values are
                # below it even after multiplying by kT
                self.freegas_cutoff = np.finfo(float).max
            else:
                self.freegas_cutoff = 400.
        else:
            self.freegas_cutoff = freegas_cutoff
        self.freegas_method = freegas_method
        self.minimum_relative_threshold = minimum_relative_threshold

    @property
    def name(self):
        return self._library.name

    @property
    def library(self):
        return self._library

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
    def fissionable(self):
        return self._fissionable

    @property
    def num_delayed_groups(self):
        return self._num_delayed_groups

    @property
    def elastic(self):
        return self._elastic

    @property
    def elastic_energy(self):
        return self._elastic_energy

    @property
    def inelastic(self):
        return self._inelastic

    @property
    def nu_inelastic(self):
        return self._nu_inelastic

    @property
    def inelastic_energy(self):
        return self._inelastic_energy

    @property
    def total_chi(self):
        return self._total_chi

    @property
    def prompt_chi(self):
        return self._prompt_chi

    @property
    def delayed_chi(self):
        return self._delayed_chi

    @property
    def chi_energy(self):
        return self._chi_energy

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
        return ["{}K".format(int(round(kT / K_BOLTZMANN))) for kT in self.kTs]

    @library.setter
    def library(self, library):
        # The library could be None when reading from disk
        cv.check_type('library', library, (IncidentNeutron, ThermalScattering,
                                           type(None)))
        self._library = library

    @group_structure.setter
    def group_structure(self, group_structure):
        cv.check_type('group_structure', group_structure,
                      EnergyGroups)
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

    @fissionable.setter
    def fissionable(self, fissionable):
        cv.check_type('fissionable', fissionable, bool)
        self._fissionable = fissionable

    @num_delayed_groups.setter
    def num_delayed_groups(self, num_delayed_groups):
        cv.check_type('num_delayed_groups', num_delayed_groups, Integral)
        self._num_delayed_groups = num_delayed_groups

    @elastic.setter
    def elastic(self, elastic):
        cv.check_type('elastic', elastic, MutableMapping)
        self._elastic = elastic

    @elastic_energy.setter
    def elastic_energy(self, elastic_energy):
        cv.check_type('elastic_energy', elastic_energy, MutableMapping)
        self._elastic_energy = elastic_energy

    @inelastic.setter
    def inelastic(self, inelastic):
        if inelastic is not None:
            cv.check_type('inelastic', inelastic, np.ndarray)
        self._inelastic = inelastic

    @nu_inelastic.setter
    def nu_inelastic(self, nu_inelastic):
        if nu_inelastic is not None:
            cv.check_type('nu_inelastic', nu_inelastic, np.ndarray)
        self._nu_inelastic = nu_inelastic

    @inelastic_energy.setter
    def inelastic_energy(self, inelastic_energy):
        if inelastic_energy is not None:
            cv.check_iterable_type('inelastic_energy', inelastic_energy, Real)
        self._inelastic_energy = inelastic_energy

    @total_chi.setter
    def total_chi(self, total_chi):
        if total_chi is not None:
            cv.check_type('total_chi', total_chi, MutableMapping)
        self._total_chi = total_chi

    @prompt_chi.setter
    def prompt_chi(self, prompt_chi):
        if prompt_chi is not None:
            cv.check_type('prompt_chi', prompt_chi, MutableMapping)
        self._prompt_chi = prompt_chi

    @delayed_chi.setter
    def delayed_chi(self, delayed_chi):
        if delayed_chi is not None:
            cv.check_type('delayed_chi', delayed_chi, MutableMapping)
        self._delayed_chi = delayed_chi

    @chi_energy.setter
    def chi_energy(self, chi_energy):
        if chi_energy is not None:
            cv.check_iterable_type('chi_energy', chi_energy, MutableMapping)
        self._chi_energy = chi_energy

    def process(self):
        """Compute the pre-processed data.
        """

        # temporarily replace the lower group boundary to a low, nonzero number
        # to avoid lots of if-branches later on
        if self.group_edges[0] == 0.:
            self.group_structure.group_edges[0] = _LOW_ENERGY
            replaced_low_energy = True
        else:
            replaced_low_energy = False

        for ikT, kT in enumerate(self.kTs):
            # Get temperature string
            strT = "{}K".format(int(round(kT / K_BOLTZMANN)))
            print('Evaluating Data at ' + strT)
            if isinstance(self.library, IncidentNeutron):
                self._compute_neutron(ikT, kT, strT)
            elif isinstance(self.library, ThermalScattering):
                self._compute_thermal(kT, strT)

        if replaced_low_energy:
            self.group_structure.group_edges[0] = 0.

    def _compute_neutron(self, ikT, kT, strT):
        """Compute the pre-processed data for IncidentNeutron data
        """

        if _FISSION_MTS.intersection(set(self.library.reactions)):
            self.fissionable = True
        else:
            self.fissionable = False

        # Calculate the scattering data
        print('\tEvaluating Elastic Data')
        self._compute_elastic(kT, strT)

        # Only do the inelastic data if it is our 1st time through since
        # it is temperature independent
        if ikT == 0:
            print('\tEvaluating Inelastic Data')
            self._compute_inelastic(kT, strT)

        # Process the fission spectra data
        if self.fissionable:
            print('\tEvaluating Fission Data')
            self._compute_chi(kT, strT)

    def _compute_thermal(self, kT, strT):
        """Computes the pre-processed data for the ThermalScattering data
        """

        # store a short hand to the library itself
        lib = self.library

        # Build the energy grid, beginning with the cross section points
        if lib.elastic_xs:
            if isinstance(lib.elastic_xs[strT], openmc.data.CoherentElastic):
                elastic_Ein = lib.elastic_xs[strT].bragg_edges
            else:
                elastic_Ein = lib.elastic_xs[strT].x

        # This method will generate the incoming energy grid according so that
        # the requested tolerance will be met. To use this we only need to
        # divide the solution space in to enough pieces that we can adequately
        # parallelize over it while also keeping in mind the desire to evenly
        # distribute the work.

        # Begin by setting the intervals to evaluate
        xs_range = [lib.inelastic_xs[strT].x[0], lib.inelastic_xs[strT].x[-1]]
        if lib.elastic_xs:
            xs_range.append(elastic_Ein[0])
            xs_range.append(elastic_Ein[-1])
        Ein_intervals = np.clip(xs_range, self.group_edges[0],
                                self.group_edges[-1])
        Ein_intervals = np.unique(Ein_intervals)
        Ein_intervals.sort()

        # Now parse through every threshold interval and, if the interval
        # has any width, add the grid points
        for e, (Ein_low, Ein_high) in enumerate(zip(Ein_intervals[:-1],
                                                    Ein_intervals[1:])):
            grid = np.logspace(np.log10(Ein_low), np.log10(Ein_high),
                               num=(_NUM_ENERGY_SUBDIV + 1), endpoint=False)
            # Add the grid to our running Ein_grid
            if e == 0:
                Ein_grid = grid
            else:
                Ein_grid = np.concatenate((Ein_grid, grid))
        # Finally add the very last interval point (since endpoint=False above)
        Ein_grid = np.concatenate((Ein_grid, [Ein_intervals[-1]]))

        # Create our mu_bins (which we do not need if doing legendre)
        if self.scatter_format == 'legendre':
            mu_bins = None
        else:
            mu_bins = np.linspace(-1, 1, self.num_angle + 1, endpoint=True)

        # Analyze the elastic grid first, if available
        if lib.elastic_xs:
            if isinstance(lib.elastic_xs[strT], openmc.data.CoherentElastic):
                elastic_Ein = lib.elastic_xs[strT].bragg_edges
                coherent = True
                mu_out = None
            else:
                elastic_Ein = lib.elastic_xs[strT].x
                coherent = False
                mu_out = lib.elastic_mu_out[strT]

            # Set up the arguments for our elastic S(a,b) integration routine
            # These have to be aggregated into a tuple to support our usage of
            # the multiprocessing pool
            elastic_args = (self.num_groups, self.num_angle,
                            self.scatter_format, self.group_edges, coherent,
                            mu_out, elastic_Ein, lib.elastic_xs[strT], mu_bins)
        else:
            elastic_args = ()

        # Do inelastic (which will be there so we dont have to check)
        if strT in lib.inelastic_dist:
            # Then we need to integrate this using the integrate(...) routine
            # more commonly used for IncidentNeutron data
            distrib = lib.inelastic_dist[strT]
            inelastic_args = (distrib, self.group_edges, self.scatter_format,
                              False, self.library.atomic_weight_ratio, 0., kT,
                              0., mu_bins, self.order, lib.inelastic_xs[strT])
            inelastic_distribution = True

        else:
            mu_data = lib.inelastic_mu_out[strT].value
            Eout_data = lib.inelastic_e_out[strT].value

            # Build weighting scheme
            wgt = np.ones(mu_data.shape[1])
            if lib.secondary_mode == 'skewed':
                wgt[0] = 0.1
                wgt[1] = 0.4
                wgt[-2] = 0.4
                wgt[-1] = 0.1
                wgt /= (np.sum(wgt) * mu_data.shape[2])
            elif lib.secondary_mode == 'equal':
                wgt /= (mu_data.shape[1] * mu_data.shape[2])

            # Set up the arguments for our inelastic S(a,b) integration routine
            # These have to be aggregated into a tuple to support our usage of
            # the multiprocessing pool
            inelastic_args = (self.num_groups, self.num_angle,
                              self.scatter_format, self.group_edges, Eout_data,
                              mu_data, mu_bins, wgt, lib.inelastic_xs[strT])
            inelastic_distribution = False

        # Set the arguments for our linearize function, except dont yet include
        # the Ein grid points (the first argument), since that will be
        # dependent upon the thread's work
        linearize_args = (do_sab, (elastic_args, inelastic_args,
                                   inelastic_distribution),
                          self.tolerance)

        inputs = [(Ein_grid[e: e + 2],) + linearize_args
                  for e in range(len(Ein_grid) - 1)]

        # Run in serial or parallel mode
        grid = [None] * len(inputs)
        results = [None] * len(inputs)
        if self.num_threads < 1:
            for e, in_data in enumerate(inputs):
                grid[e], results[e] = linearizer_wrapper(in_data)
        else:
            p = Pool(self.num_threads)
            output = p.map(linearizer_wrapper, inputs)
            p.close()
            # output contains a 2-tuple for every parallelized bin;
            # the first entry in the tuple is the energy grid, and the second
            # is the results array. We need to separate and combine these
            for e in range(len(inputs)):
                grid[e] = output[e][0]
                results[e] = output[e][1]

        # Now lets combine our grids together
        Ein_grid = np.concatenate(grid)
        results = np.concatenate(results)

        # Remove the non-unique entries obtained since the linearizer
        # includes the endpoints of each bracketed region
        Ein_grid, unique_indices = np.unique(Ein_grid, return_index=True)
        results = results[unique_indices, ...]

        # Normalize results to remove the cross section dependence and
        # any numerical errors
        if self.scatter_format == 'legendre':
            divisor = np.sum(results[:, :, 0], axis=1)
        else:
            divisor = np.sum(results[:, :, :], axis=(1, 2))
        for i in range(len(Ein_grid)):
            if divisor[i] > 0.:
                results[i, :, :] = \
                    np.divide(results[i, :, :], divisor[i])

        self.elastic_energy[strT] = Ein_grid
        self.elastic[strT] = results

    def _compute_elastic(self, kT, strT):
        """Computes the pre-processed energy-angle data from the elastic
        scattering reaction of an IncidentNeutron dataset.
        """

        # Get our reaction data
        awr = self.library.atomic_weight_ratio

        # Pre-calculate our free-gas kernel weights if needed
        if self.scatter_format == 'legendre':
            mus_grid, wgts = initialize_quadrature(self.order)
        else:
            mus_grid, wgts = None, None

        # Create our mu_bins (which we do not need if doing Legendre)
        if self.scatter_format == 'legendre':
            mu_bins = None
        else:
            mu_bins = np.linspace(-1, 1, self.num_angle + 1, endpoint=True)

        # This method will generate the incoming energy grid according so that
        # the requested tolerance will be met. To use this we only need to
        # divide the solution space in to enough pieces that we can adequately
        # parallelize over it while also keeping in mind the desire to evenly
        # distribute the work.
        # For elastic scattering, this means making sure all threads are
        # working on chunks of the free gas portion since that is very time
        # intensive.
        Ein_low = self.group_edges[0]
        Ein_high = self.group_edges[-1]
        # Reduce the high Ein if we need to consider freegas scattering
        if self.freegas_cutoff > 0.:
            Ein_high = min(self.freegas_cutoff * kT, self.group_edges[-1])
        Ein_grid = np.logspace(np.log10(Ein_low), np.log10(Ein_high),
                               num=(_NUM_ENERGY_SUBDIV),
                               endpoint=False)

        # If Ein_high was set by the freegas cutoff, then we need to
        # also add in the bins for the non-freegas part
        if Ein_high == self.freegas_cutoff * kT:
            Ein_low = Ein_high
            Ein_high = self.group_edges[-1]
            upper_grid = np.logspace(np.log10(Ein_low),
                                     np.log10(Ein_high),
                                     num=(_NUM_ENERGY_SUBDIV + 1),
                                     endpoint=True)
        else:
            # Since the first logspace call had endpoint=False, and we now
            # want that endpoint, lets add it in
            upper_grid = [self.group_edges[-1]]
        Ein_grid = np.concatenate((Ein_grid, upper_grid))

        # Now add in the critical points (Ein where Eout reaches a group
        # threshold); this will be done not considering the free-gas treatment
        crit_pts = []
        for Eout in self.group_edges[:-1]:
            crit_pts.append(Eout * (1. + awr)**2 / (awr * awr + 2. * awr + 1.))
            crit_pts.append(Eout * (1. + awr)**2 / (awr * awr - 2. * awr + 1.))
        crit_pts = np.sort(crit_pts)
        crit_pts = np.clip(crit_pts, self.group_edges[0], self.group_edges[-1])
        Ein_grid = np.union1d(Ein_grid, crit_pts)

        Ein_grid = np.unique(Ein_grid)

        # We want our lower Eout bound to be 0 for these integrations, so
        # lets momentarily switch the group structure to have a 0 value
        # at the bottom
        if self.group_edges[0] == _LOW_ENERGY:
            self.group_structure.group_edges[0] = 0.

        # Set up the arguments for our do_neutron_scatter routine
        # These have to be aggregated into a tuple to support our usage of
        # the multiprocessing pool
        func_args = (awr, self.library.reactions[2],
                     self.library.reactions[2].products[0], self, kT, mu_bins,
                     self.library.reactions[2].xs[strT], mus_grid, wgts)

        # Set the free-gas method to use
        set_freegas_method(self.freegas_method == 'cxs')

        # Set the arguments for our linearize function, except dont yet include
        # the Ein grid points (the first argument), since that will be
        # dependent upon the thread's work
        linearize_args = (do_by_rxn, func_args, self.tolerance)

        inputs = [(Ein_grid[e: e + 2],) + linearize_args
                  for e in range(len(Ein_grid) - 1)]

        # Run in serial or parallel mode
        grid = [None] * len(inputs)
        results = [None] * len(inputs)
        if self.num_threads < 1:
            for e, in_data in enumerate(inputs):
                grid[e], results[e] = linearizer_wrapper(in_data)
        else:
            p = Pool(self.num_threads)
            output = p.map(linearizer_wrapper, inputs)
            p.close()
            # output contains a 2-tuple for every parallelized bin;
            # the first entry in the tuple is the energy grid, and the second
            # is the results array. We need to separate and combine these
            for e in range(len(inputs)):
                grid[e] = output[e][0]
                results[e] = output[e][1]

        # Reverse what we did to the group structure
        if self.group_edges[0] == 0.:
            self.group_structure.group_edges[0] = _LOW_ENERGY

        # Now lets combine our grids together
        Ein_grid = np.concatenate(grid)
        results = np.concatenate(results)

        # Remove the non-unique entries obtained since the linearizer
        # includes the endpoints of each bracketed region
        Ein_grid, unique_indices = np.unique(Ein_grid, return_index=True)
        self.elastic[strT] = results[unique_indices, ...]

        # Normalize results to remove the cross section dependence and
        # any numerical errors
        if self.scatter_format == 'legendre':
            divisor = np.sum(self.elastic[strT][:, :, 0], axis=1)
        else:
            divisor = np.sum(self.elastic[strT][:, :, :], axis=(1, 2))
        for e in range(len(Ein_grid)):
            if divisor[e] > 0.:
                self.elastic[strT][e, :, :] = \
                    np.divide(self.elastic[strT][e, :, :], divisor[e])

        # Finally add a top point to use as interpolation
        Ein_grid = np.append(Ein_grid, Ein_grid[-1] + 1.e-1)
        self.elastic_energy[strT] = Ein_grid
        self.elastic[strT] = np.concatenate((self.elastic[strT],
                                             [self.elastic[strT][-1, :, :]]))

    def _compute_inelastic(self, kT, strT):
        """Computes the pre-processed energy-angle data from the inelastic
        scattering reactions of an IncidentNeutron dataset.
        """

        # store a short hand to the library itself and the atomic weight ratio
        awr = self.library.atomic_weight_ratio

        # Get the relevant reactions, their products, and their products'
        # distributions
        # We will also find the minimum threshold energy of these reactions
        Emin = self.group_edges[-1]
        rxns = []
        products = []
        xs_funcs = []
        thresholds = []
        for r in _INELASTIC_MTS.intersection(set(self.library.reactions)):
            rxn = self.library.reactions[r]
            include = False
            for p in rxn.products:
                if (p.particle == 'neutron' and p.emission_mode == 'prompt') \
                    and rxn.xs[strT].x[0] < self.group_edges[-1]:

                    products.append(p)
                    include = True
                    continue
            if include:
                Emin = min(Emin, rxn.xs[strT].x[0])
                rxns.append(rxn)
                xs_funcs.append(rxn.xs[strT])
                thresholds.append(rxn.xs[strT].x[0])

        # If we found no inelastic data, dont waste time with the rest
        if not rxns:
            return

        # Pre-calculate our quadrature integration weights if needed
        if self.scatter_format == 'legendre':
            mus_grid, wgts = initialize_quadrature(self.order)
        else:
            mus_grid, wgts = None, None

        # Create our mu_bins (which we do not need if doing Legendre)
        if self.scatter_format == 'legendre':
            mu_bins = None
        else:
            mu_bins = np.linspace(-1, 1, self.num_angle + 1, endpoint=True)

        # Loop over incoming groups so we dont have too much memory storage
        # for each reaction at once before we combine the data
        for g in range(self.num_groups):
            print("\t  Analyzing Incoming Energy Group {}".format(
                  self.num_groups - g))
            # We will parse through each reaction one at a time. Doing so
            # allows us to get the variation in the reaction-wise distributions
            # themselves without having to individually evaluate the variation
            # in yields and xs
            # Once we have the distribution variations we will combine on to a
            # unionized inelastic energy grid
            rxn_results = [None] * len(rxns)
            rxn_grids = [None] * len(rxns)
            above_threshold = False

            for r, rxn in enumerate(rxns):
                if thresholds[r] > self.group_edges[g + 1]:
                    # Then there is no reason to compute anything, move on
                    continue

                # Note that we actually were above the threshold
                above_threshold = True

                print('\t\t', rxn)
                # Create the energy grid we will be evaluating for this case
                Ein_grid = np.logspace(np.log10(max(thresholds[r],
                                                    self.group_edges[g])),
                                       np.log10(self.group_edges[g + 1]),
                                       num=_NUM_ENERGY_SUBDIV + 1,
                                       endpoint=True)

                # Set up the arguments for our processing routine
                # These have to be aggregated into a tuple to support the usage
                # of the multiprocessing pool
                func_args = (awr, rxn, products[r], self, kT, mu_bins,
                             xs_funcs[r], mus_grid, wgts)

                # Set the arguments for our linearize function, except dont yet
                # include the Ein grid points (the first argument), since that
                # will be dependent upon the thread's given work
                linearize_args = (do_by_rxn, func_args, self.tolerance)

                inputs = [(Ein_grid[e: e + 2],) + linearize_args
                          for e in range(len(Ein_grid) - 1)]

                # Run in serial or parallel mode
                grid = [None] * len(inputs)
                results = [None] * len(inputs)
                if self.num_threads < 1:
                    for e, in_data in enumerate(inputs):
                        grid[e], results[e] = linearizer_wrapper(in_data)
                else:
                    p = Pool(self.num_threads)
                    output = p.map(linearizer_wrapper, inputs)
                    p.close()
                    # output contains a 2-tuple for every parallelized bin;
                    # the first entry in the tuple is the energy grid, and the
                    # 2nd is the results array. We need to separate and combine
                    # these
                    for e in range(len(inputs)):
                        grid[e] = output[e][0]
                        results[e] = output[e][1]

                # Now lets combine our grids together
                rxn_grids[r] = np.concatenate(grid)
                rxn_results[r] = np.concatenate(results)

                # Retain only the unique end-points
                rxn_grids[r], unique_indices = np.unique(rxn_grids[r],
                                                         return_index=True)
                rxn_results[r] = rxn_results[r][unique_indices, ...]

            # Now combine the data on to a unionized energy grid
            # The inelastic data can have many points and so we have to do this
            # in a memory conscious way.
            # So, we will attack the problem by thinning according to the
            # user's chosen tolerance as we go

            # Make sure we had any data above the threshold
            if not above_threshold:
                continue

            print("\t\t Combining Reactions")

            # Start with the energy grid
            Ein_grid = []
            for grid in rxn_grids:
                if grid is not None:
                    Ein_grid = np.union1d(Ein_grid, grid)

            new_Ein_grid = []
            nu_inelastic = []

            # Now step through each point, compare with the neighbors and see
            # if it is necessary
            for e, Ein in enumerate(Ein_grid):
                combined = np.zeros((self.num_groups, self.num_angle))
                for r in range(len(rxn_grids)):
                    if Ein > thresholds[r]:
                        # Find the corresponding point
                        if Ein < rxn_grids[r][-2]:
                            i = np.searchsorted(rxn_grids[r], Ein) - 1
                        else:
                            i = len(rxn_grids[r]) - 2

                        # Get the interpolant
                        f = (Ein - rxn_grids[r][i]) / \
                            (rxn_grids[r][i + 1] - rxn_grids[r][i])

                        combined += (1. - f) * rxn_results[r][i, ...] + \
                            f * rxn_results[r][i + 1, ...]

                combined = sps.csr_matrix(combined)

                if len(nu_inelastic) >= 2:
                    # If we have enough points already, use our latest point,
                    # in combined, and the point calculated (and kept) 2 times
                    # ago, and see if the point calculated (and kept)
                    # immediately prior to this is useful

                    # Find the interpolable result
                    f = (new_Ein_grid[-1] - new_Ein_grid[-2]) / \
                        (Ein - new_Ein_grid[-2])
                    results_interp = (1. - f) * nu_inelastic[-2] + f * combined

                    # Now see if the interpolable result is within our
                    # tolerance; if it is, keep it.
                    error = np.subtract(results_interp, nu_inelastic[-1])
                    # Avoid division by 0 errors since they are fully expected
                    # with our sparse results
                    with np.errstate(divide='ignore', invalid='ignore'):
                        error = \
                            np.abs(np.nan_to_num(np.divide(error,
                                                           nu_inelastic[-1])))

                    if np.all(error < self.tolerance):
                        # If the error is sufficiently small, the middle point
                        # is not needed
                        del new_Ein_grid[-1]
                        del nu_inelastic[-1]

                # Add the point we just calculated, unless this is the last
                # point in the set.
                # However, for the highest energy group, I do want that point
                if (e < len(Ein_grid) - 1) or (g == self.num_groups - 1):
                    new_Ein_grid.append(Ein)

                    # And add combined, and when we do it, lets save space by
                    # using a sparse representation
                    nu_inelastic.append(sps.csr_matrix(combined))

            # And add to our growing list of energies and results
            if self.inelastic_energy is not None:
                self.inelastic_energy = np.concatenate((self.inelastic_energy,
                                                        new_Ein_grid))
                # self.nu_inelastic = np.concatenate((self.nu_inelastic,
                #                                     nu_inelastic))
                self.nu_inelastic = np.append(self.nu_inelastic, nu_inelastic)
            else:
                # First time through, nothing to append to
                self.inelastic_energy = np.array(new_Ein_grid)
                self.nu_inelastic = np.array(nu_inelastic)
            print(len(self.inelastic_energy))

        # Convert the nu_inelastic data into inelastic data, we will do this by
        # normalizing nu_inelastic and scale by the inelastic_xs
        inelastic_xs = get_inelastic_xs(self.inelastic_energy, rxns, xs_funcs)
        inelastic = np.zeros_like(self.nu_inelastic)
        for e in range(len(self.inelastic_energy)):
            if self.scatter_format == 'legendre':
                norm = np.sum(self.nu_inelastic[e][:, 0])
            else:
                norm = np.sum(self.nu_inelastic[e][:, :])

            if norm > 0.:
                inelastic[e] = self.nu_inelastic[e] / norm * inelastic_xs[e]
            else:
                inelastic[e] = self.nu_inelastic[e]
        # if self.scatter_format == 'legendre':
        #     norm = np.sum(self.nu_inelastic[:, :, 0], axis=1)
        # else:
        #     norm = np.sum(self.nu_inelastic, axis=(1, 2))

        # for e in range(len(self.inelastic_energy)):
        #     if norm[e] > 0.:
        #         inelastic[e] = \
        #             self.nu_inelastic[e, :, :] / norm[e] * inelastic_xs[e]
        self.inelastic = inelastic

        # Finally, duplicate the top point to help with interpolation if Ein is
        # exactly equal in the Monte Carlo code to the top energy
        self.inelastic_energy = np.append(self.inelastic_energy,
                                          self.inelastic_energy[-1] + 1.e-1)
        # self.nu_inelastic = np.concatenate((self.nu_inelastic,
        #                                     [self.nu_inelastic[-1, :, :]]))
        # self.inelastic = np.concatenate((self.inelastic,
        #                                  [self.inelastic[-1, :, :]]))
        self.nu_inelastic = np.concatenate((self.nu_inelastic,
                                            [self.nu_inelastic[-1]]))
        self.inelastic = np.concatenate((self.inelastic,
                                         [self.inelastic[-1]]))

    def _compute_chi(self, kT, strT):
        """Computes the pre-processed fission spectral data from an
        IncidentNeutron dataset.
        """

        # Build the incoming energy grids.
        # We also will gather references to the per-rxn information we will
        # later need for processing.
        rxns = []
        xs_funcs = []
        thresholds = []

        n_delayed = 0
        for r in _FISSION_MTS.intersection(set(self.library.reactions)):
            rxn = self.library.reactions[r]
            # Store the reaction and create new lists to traverse for the
            # products, distributions, and applicability indices
            rxns.append(rxn)
            xs_funcs.append(rxn.xs[strT])
            thresholds.append(rxn.xs[strT].x[0])

            # Look in all the products for distributions of interest so we can
            # count the number of delayed precursor groups
            for p in (rxn.derived_products + rxn.products):
                if p.particle == 'neutron' and p.emission_mode == 'delayed':
                    n_delayed += 1

        # If we found no fission data, dont waste time with the rest
        if not rxns:
            return

        # We will parse through each relevant reaction one at a time.
        # Doing so allows us to get the variation in the reaction-wise
        # distributions themselves without having to individually evaluate
        # their variation in yields and xs
        # Since we know the fission x/s are likely strongly varying (since
        # these reactions have been measured and evaluated heavily, unlike
        # inelastic scatter channels), we will put the data on to the x/s
        # grid, normalize the CDF to 1.0, and then make sure the grid
        # is thinned so we arent giving the downstream code way to much data.

        rxn_results = [None] * len(rxns)
        rxn_grids = [None] * len(rxns)
        for r, rxn in enumerate(rxns):
            print('\t\t', rxn)

            # Create the energy grid we will be evaluating for this case
            Ein_grid = np.logspace(np.log10(max(thresholds[r],
                                                self.group_edges[0])),
                                   np.log10(self.group_edges[-1]),
                                   num=_NUM_ENERGY_SUBDIV + 1)

            # Set up the arguments for our do_chi routine
            # These have to be aggregated into a tuple to support our usage of
            # the multiprocessing pool
            func_args = (self.library.atomic_weight_ratio, rxn, self, kT,
                         xs_funcs[r], n_delayed)

            # Set the arguments for our linearize function, except dont yet
            # include the Ein grid points (the first argument), since that will
            # be dependent upon the thread's work
            linearize_args = (do_chi, func_args, self.tolerance)

            inputs = [(Ein_grid[e: e + 2],) + linearize_args
                      for e in range(len(Ein_grid) - 1)]

            # Run in serial or parallel mode
            grid = [None] * len(inputs)
            results = [None] * len(inputs)
            if self.num_threads < 1:
                for e, in_data in enumerate(inputs):
                    grid[e], results[e] = linearizer_wrapper(in_data)
            else:
                p = Pool(self.num_threads)
                output = p.map(linearizer_wrapper, inputs)
                p.close()
                # output contains a 2-tuple for every parallelized bin;
                # the 1st entry in the tuple is the energy grid, and the 2nd
                # is the results array. We need to separate and combine these
                for e in range(len(inputs)):
                    grid[e] = output[e][0]
                    results[e] = output[e][1]

            # Now lets combine our grids together
            rxn_grids[r] = np.concatenate(grid)
            rxn_results[r] = np.concatenate(results)

            # Retain only the unique end-points
            rxn_grids[r], unique_indices = np.unique(rxn_grids[r],
                                                     return_index=True)
            rxn_results[r] = rxn_results[r][unique_indices, ...]

        if len(rxns) > 1:
            print("\t\t Combining Reactions")
            # Then we have to combine the results on to a unionized grid
            # This unionized grid will be the union of the rxn_grids above
            # and the actual x/s grid in the range of interest

            # First lets get the x/s grid in the range of interest
            # We dont need to combine all x/s grids, since they are all the
            # same. We will also use the xs data from the first entry in rxns
            # and xs_funcs, since that will be the MT=18 channel
            xs_grid = np.unique(np.clip(xs_funcs[0]._x, self.group_edges[0],
                                        self.group_edges[-1]))

            # Combine the reaction grids
            Ein_grid = reduce(np.union1d, rxn_grids)

            # Finally combine the reaction grid and xs grid
            Ein_grid = np.union1d(Ein_grid, xs_grid)

            pd_results = np.zeros((len(Ein_grid),) + rxn_results[0].shape[1:])

            # Now interpolate the reaction distributions at each Ein on xs_grid
            # and combine in to a unionized grid
            for e, Ein in enumerate(Ein_grid):
                for r in range(len(rxn_grids)):
                    if Ein > thresholds[r]:
                        # Find the corresponding point
                        if Ein < rxn_grids[r][-2]:
                            i = np.searchsorted(rxn_grids[r], Ein) - 1
                        else:
                            i = len(rxn_grids[r]) - 2

                        # Get the interpolant
                        f = (Ein - rxn_grids[r][i]) / \
                            (rxn_grids[r][i + 1] - rxn_grids[r][i])

                        pd_results[e, ...] += \
                            ((1. - f) * rxn_results[r][i, ...] +
                             f * rxn_results[r][i + 1, ...]) * xs_funcs[r](Ein)

        else:
            # If we only have the total fission channel, then we dont have to
            # worry about x/s interference and we can instead skip right to the
            # normalization step
            # Lets just put the data in to the same variables as the other
            # side of the if-branch
            Ein_grid = rxn_grids[0]
            pd_results = rxn_results[0]

        # Create the total secondary energy data (i.e. prompt + delayed)
        tot_results = np.sum(pd_results, axis=1)

        # With chi data, we only care about the normalized secondary energy
        # data, so lets normalize before we thin
        tot_results[:, :] = np.nan_to_num(
            np.divide(tot_results[:, :],
                      np.sum(tot_results[:, :], axis=1)[:, None]))
        pd_results[:, :, :] = np.nan_to_num(
            np.divide(pd_results, np.sum(pd_results, axis=-1)[:, :, None]))

        # Add a top point to use as interpolation
        Ein_grid = np.append(Ein_grid, Ein_grid[-1] + 1.e-1)
        pd_results = np.concatenate((pd_results, [pd_results[-1, :, :]]))
        tot_results = np.concatenate((tot_results, [tot_results[-1, :]]))

        # Finally store the incoming energy grid and chi values
        self.chi_energy[strT] = Ein_grid[:]
        self.prompt_chi[strT] = pd_results[:, 0, :]
        self.delayed_chi[strT] = pd_results[:, 1:, :]
        self.total_chi[strT] = tot_results[:, :]
        self.num_delayed_groups = n_delayed

    def to_hdf5(self, file):
        """Export processed data to an HDF5 file.

        Parameters
        ----------
        file : h5py.File
            HDF5 File (a root Group) to write to

        """

        # Open file and write version
        g = file.create_group(self.name)
        g.attrs['fissionable'] = (self.fissionable)
        g.attrs['num_delayed_groups'] = self.num_delayed_groups
        ktg = g.create_group('kTs')
        for i, temperature in enumerate(self.temperatures):
            ktg.create_dataset(temperature, data=self.kTs[i])

        # Add temperature dependent data
        for T, energy in self.elastic_energy.items():
            g_out_bounds, flattened = _sparsify(energy, self.elastic[T],
                                                self.scatter_format,
                                                self.minimum_relative_threshold)
            Tgroup = g.create_group(T)
            Tgroup.create_dataset('elastic_energy', data=energy)
            Tgroup.create_dataset('elastic', data=flattened)
            Tgroup.create_dataset("elastic_g_min", data=g_out_bounds[:, 0])
            Tgroup.create_dataset("elastic_g_max", data=g_out_bounds[:, 1])

        # Temperature independent data
        if self.inelastic_energy is not None:
            g_out_bounds, flattened = _sparsify2(self.inelastic_energy,
                                                self.inelastic,
                                                self.scatter_format,
                                                self.minimum_relative_threshold)
            g.create_dataset('inelastic_energy', data=self.inelastic_energy)
            g.create_dataset('inelastic', data=flattened)
            g.create_dataset("inelastic_g_min", data=g_out_bounds[:, 0])
            g.create_dataset("inelastic_g_max", data=g_out_bounds[:, 1])
            g_out_bounds, flattened = _sparsify2(self.inelastic_energy,
                                                self.nu_inelastic,
                                                self.scatter_format,
                                                self.minimum_relative_threshold)
            g.create_dataset('nu_inelastic', data=flattened)

        if self.chi_energy:
            for T, energy in self.chi_energy.items():
                Tgroup = g[T]
                Tgroup.create_dataset('chi_energy', data=energy)
                g_out_bounds, flattened = \
                    _sparsify(energy, self.total_chi[T], 'chi',
                              self.minimum_relative_threshold)
                Tgroup.create_dataset('total_chi', data=flattened)
                g_out_bounds, flattened = \
                    _sparsify(energy, self.prompt_chi[T], 'chi',
                              self.minimum_relative_threshold)
                Tgroup.create_dataset("total_chi_g_min",
                                      data=g_out_bounds[:, 0])
                Tgroup.create_dataset("total_chi_g_max",
                                      data=g_out_bounds[:, 1])
                Tgroup.create_dataset('prompt_chi', data=flattened)
                Tgroup.create_dataset("prompt_chi_g_min",
                                      data=g_out_bounds[:, 0])
                Tgroup.create_dataset("prompt_chi_g_max",
                                      data=g_out_bounds[:, 1])
                dgroup = Tgroup.create_group('delayed_chi')
                for c in range(self.num_delayed_groups):
                    g_out_bounds, flattened = \
                        _sparsify(energy, self.delayed_chi[T][:, c, :], 'chi',
                                  self.minimum_relative_threshold)
                    dgroup.create_dataset(str(c + 1), data=flattened)
                    dgroup.create_dataset(str(c + 1) + "_g_min",
                                          data=g_out_bounds[:, 0])
                    dgroup.create_dataset(str(c + 1) + "_g_max",
                                          data=g_out_bounds[:, 1])

    @classmethod
    def from_hdf5(cls, group_or_filename):
        """Generate continuous-energy neutron interaction data from HDF5 group

        Parameters
        ----------
        group_or_filename : h5py.Group or str
            HDF5 group containing the pre-processed data. If given as a string,
            it is assumed to be the filename for the HDF5 file, and the first
            group is used to read from.

        Returns
        -------
        openmc.data.Ndpp
            Pre-processed nuclear data (NDPP) object

        """
        if isinstance(group_or_filename, h5py.Group):
            group = group_or_filename
        else:
            h5file = h5py.File(group_or_filename, 'r')

            # Make sure version matches
            if 'version' in h5file.attrs:
                major, minor = h5file.attrs['version']
                if major != NDPP_VERSION_MAJOR:
                    raise IOError(
                        'HDF5 data format uses version {}.{} whereas your '
                        'installation of the OpenMC Python API expects '
                        ' version{}.x.'.format(major, minor,
                                               NDPP_VERSION_MAJOR))
            else:
                raise IOError(
                    'HDF5 data does not indicate a version. Your installation '
                    'of the OpenMC Python API expects version {}.x data.'
                    .format(NDPP_VERSION_MAJOR))

            group = list(h5file.values())[0]

        group_structure = \
            EnergyGroups(group.attrs['group_structure'])
        scatter_format = group.attrs['scatter_format']
        order = group.attrs['order']
        freegas_cutoff = group.attrs['freegas_cutoff']
        freegas_method = group.attrs['freegas_method']
        library = None
        if freegas_cutoff == -1:
            freegas_cutoff = None

        kTg = group['kTs']
        kTs = []
        for temp in kTg:
            kTs.append(kTg[temp].value)

        data = cls(library, group_structure, scatter_format, order, kTs=kTs,
                   freegas_cutoff=freegas_cutoff,
                   freegas_method=freegas_method)

        # Read temperature dependent data
        for T, Tgroup in group.items():
            if T.endswith('K'):
                grid = Tgroup['elastic_energy'].value
                data.elastic_energy[T] = grid[:]
                data = Tgroup['elastic'].value
                data.elastic_energy[T] = data[:, :, :]
                if 'chi_energy' in Tgroup:
                    data.fissionable = True
                    data.chi_energy[T] = group['chi_energy'].value[:]
                    data.total_chi[T] = group['total_chi'].value[:, :]
                    data.prompt_chi[T] = group['prompt_chi'].value[:, :]
                    data.delayed_chi[T] = group['delayed_chi'].value[:, :, :]

        # Read temperature independent data
        if 'inelastic_energy' in group:
            data.inelastic_energy = group['inelastic_energy'].value[:]
            data.inelastic = group['inelastic'].value[:]
            data.nu_inelastic = group['nu_inelastic'].value[:]

        return data


def get_inelastic_xs(Eins, rxns, xs_func):
    inelastic_xs = np.zeros_like(Eins)
    # Get the cross section for each reaction/product pair
    for r, rxn in enumerate(rxns):
        xs = xs_func[r]
        for e, Ein in enumerate(Eins):
            if Ein > xs._x[0]:
                inelastic_xs[e] += xs(Ein)

    return inelastic_xs


def initialize_quadrature(order):
    # Set our lab-frame angles to calculate at and calculate the weights
    # of the quadrature integration
    # N is the quadrature integration order and K is the number of Lagrange
    # points

    # First find our K zeros
    # Set up the polynomial so we can find our poly.
    coeffs = [0] * (_K_LAGRANGE + 1)
    coeffs[-1] = 1
    mus_k = np.polynomial.legendre.legroots(coeffs)
    mus_k = np.ascontiguousarray(mus_k)

    # Now we need to find how many orders (N) we need to include in our
    # quadrature integration
    N = int(np.ceil((_K_LAGRANGE + order - 1) / 2))

    # With our N, find the quadrature points
    mus_n, wgts_n = np.polynomial.legendre.leggauss(N)
    mus_n = np.ascontiguousarray(mus_n)
    wgts_n = np.ascontiguousarray(wgts_n)

    # Now we have to find the values of \alpha_{k,n}, which is equal to
    # the lagrange interpolating function evaluated at the quadrature
    # points.
    # First define the interpolationg function
    def f_k(mu, k, mus_k):
        # mu is the value to to evaluate at
        # k is the order of the lagrange function
        # mus_k is a list of our interpolant points
        num = 1.
        denom = 1.
        for i in range(len(mus_k)):
            if i != k:
                num *= (mu - mus_k[i])
                denom *= (mus_k[k] - mus_k[i])
        return num / denom

    # Find \alpha_{k,n} all we need to do is evaluate f_k at mus_n
    alpha_kn = np.empty((_K_LAGRANGE, len(mus_n)), np.float)
    for k in range(_K_LAGRANGE):
        for n in range(N):
            alpha_kn[k, n] = f_k(mus_n[n], k, mus_k)

    # Find the weights to use for each scattering moment Legendre order
    # at every quadrature integration point
    wgts_ln = np.empty((order + 1, N), np.float)
    for l in range(order + 1):
        for n in range(N):
            wgts_ln[l, n] = wgts_n[n] * ss.eval_legendre(l, mus_n[n])

    # Finally, to save expense later, lets now calculate
    # wgts = \sum_{n} w_{l,n} * \alpha_{k, n}
    wgts = np.empty((_K_LAGRANGE, order + 1), np.float)
    for k in range(_K_LAGRANGE):
        for l in range(order + 1):
            wgts[k, l] = np.dot(wgts_ln[l, :], alpha_kn[k, :])

    return mus_k, wgts


def _sparsify(energy, data, datatype, min_rel_threshold):
    g_out_bounds = np.zeros((len(energy), 2), dtype=np.int)
    for ei in range(len(energy)):
        if datatype == 'legendre':
            matrix = data[ei, :, 0]
        elif datatype == 'histogram':
            matrix = \
                np.sum(data[ei, :, :],
                       axis=1)
        elif datatype == 'chi':
            matrix = data[ei, :]

        # Apply the relative threshold
        thresh = np.sum(matrix) * min_rel_threshold
        matrix[matrix < thresh] = 0.

        nz = np.nonzero(matrix)
        # It is possible that there only zeros in matrix
        # and therefore nz will be empty, in that case set
        # g_out_bounds to -1s
        if len(nz[0]) == 0:
            g_out_bounds[ei, :] = -1
        else:
            g_out_bounds[ei, 0] = nz[0][0]
            g_out_bounds[ei, 1] = nz[0][-1]

    # Now create the flattened array
    flattened = []
    for ei in range(len(energy)):
        if datatype is not 'chi':
            for g_out in range(g_out_bounds[ei, 0], g_out_bounds[ei, 1] + 1):
                for l in range(len(data[ei, g_out, :])):
                    flattened.append(data[ei, g_out, l])
        else:
            for g_out in range(g_out_bounds[ei, 0], g_out_bounds[ei, 1] + 1):
                flattened.append(data[ei, g_out])

    # And finally, adjust g_out_bounds for 1-based group counting
    g_out_bounds[:, :] += 1

    return g_out_bounds, np.array(flattened)


def _sparsify2(energy, data, datatype, min_rel_threshold):
    g_out_bounds = np.zeros((len(energy), 2), dtype=np.int)
    for ei in range(len(energy)):
        if datatype == 'legendre':
            matrix = data[ei].toarray()[:, 0]
        elif datatype == 'histogram':
            matrix = \
                np.sum(data[ei].toarray()[:, :],
                       axis=1)
        elif datatype == 'chi':
            matrix = data[ei].toarray()[:]

        # Apply the relative threshold
        thresh = np.sum(matrix) * min_rel_threshold
        matrix[matrix < thresh] = 0.

        nz = np.nonzero(matrix)
        # It is possible that there only zeros in matrix
        # and therefore nz will be empty, in that case set
        # g_out_bounds to -1s
        if len(nz[0]) == 0:
            g_out_bounds[ei, :] = -1
        else:
            g_out_bounds[ei, 0] = nz[0][0]
            g_out_bounds[ei, 1] = nz[0][-1]

    # Now create the flattened array
    flattened = []
    for ei in range(len(energy)):
        if datatype is not 'chi':
            for g_out in range(g_out_bounds[ei, 0], g_out_bounds[ei, 1] + 1):
                for l in range(len(data[ei].toarray()[g_out, :])):
                    flattened.append(data[ei].toarray()[g_out, l])
        else:
            for g_out in range(g_out_bounds[ei, 0], g_out_bounds[ei, 1] + 1):
                flattened.append(data[ei].toarray()[g_out])

    # And finally, adjust g_out_bounds for 1-based group counting
    g_out_bounds[:, :] += 1

    return g_out_bounds, np.array(flattened)
