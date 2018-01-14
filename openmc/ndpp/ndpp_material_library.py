from collections import Iterable

import numpy as np

from openmc.data import K_BOLTZMANN
import openmc.checkvalue as cv
from .evaluators import *
from .ndpp import Ndpp
from .sparsescatter import *
from .ndpplibrary import NdppLibrary


class NdppMaterialLibrary(NdppLibrary):
    """Pre-processed nuclear data file used for improved tallying efficiency
    in an OpenMC simulation; this is class is specific to macroscopic
    (i.e., material-wise) data instead of nuclidic data.

    Parameters
    ----------
    model : openmc.Model
        Model information used to create the materials
    nuclide_ndpp_path : str
        Path to read the nuclidic NDPP HDF5 file
    material_ndpp_path : str
        Path to write the material NDPP HDF5 file; defaults to
        'macro_ndpp_lib.h5'

    Attributes
    ----------
    names : List of str
        The names of all the materials contained within this NDPP Library
    materials : dict of openmc.Material
        The list of materials included in this data library; the key of the
        dict is the index of that material in materials
    material_temperatures : dict of float
        This dictioanry contains the temperatures that each material is
        provided at; the key of the dict is the index of that material in
        materials
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
        Pre-processed data for each material

    """

    def __init__(self, model, nuclide_ndpp_path):
        """Create an hdf5 file with NDPP objects specific to the material data
        itself.

        Parameters
        ----------
        model : openmc.Model
            Model information used to create the materials
        nuclide_ndpp_path : str
            Path to read nuclidic NDPP HDF5 file
        """

        # Open the nuclidic library
        nuclide_ndpp_lib = NdppLibrary.from_hdf5(nuclide_ndpp_path)

        # Set parameters from the nuclidic library
        self.cross_sections_xml = nuclide_ndpp_lib.cross_sections_xml
        self.group_structure = nuclide_ndpp_lib.group_structure
        self.num_threads = nuclide_ndpp_lib.num_threads
        self.tolerance = nuclide_ndpp_lib.tolerance
        self.minimum_relative_threshold = \
            nuclide_ndpp_lib.minimum_relative_threshold
        self.scatter_format = nuclide_ndpp_lib.scatter_format
        self.order = nuclide_ndpp_lib.order
        self.freegas_cutoff = nuclide_ndpp_lib.freegas_cutoff
        self.freegas_method = nuclide_ndpp_lib.freegas_method

        # Set the temperatures to None since there wont be a library-wide
        # temperature set that all are evaluated at
        self.kTs = None

        # Get default temperature
        if model.settings.temperature:
            default_T = model.settings.temperature['default']
        else:
            default_T = 293.6

        materials = model.geometry.get_all_materials()
        # Create a dict of temperature sets for each material
        temperatures = {}
        for mat_id in materials:
            temperatures[mat_id] = []

            # If the material has its own temperature, add it to the set
            if materials[mat_id].temperature is not None:
                temperatures[mat_id].append(materials[mat_id].temperature)

        # Now go through the cells and find other temperatures for the material
        for cell in model.geometry.get_all_cells().values():
            cell_materials = cell.get_all_materials()
            for mat_id in cell_materials:
                if isinstance(cell.temperature, Iterable):
                    temperatures[mat_id].extend(cell.temperature)
                elif cell.temperature is not None:
                    temperatures[mat_id].append(cell.temperature)

        # Now check material temperatures; if any are empty, then set to the
        # default temperature
        for mat_id in materials:
            if not temperatures[mat_id]:
                temperatures[mat_id].append(default_T)

        # Save the material and temperature information
        self.materials = materials

        self.ndpps = []
        for mat_id in self.materials:
            self.ndpps.append(NdppMaterial(self.materials[mat_id],
                                           temperatures[mat_id],
                                           nuclide_ndpp_lib))


class NdppMaterial(Ndpp):
    """This class takes previously processed
    nuclidic NDPP data and combines this microscopic information into
    macroscopic information.

    Parameters
    ----------
    material : openmc.Material
        The material included in this data library
    temperatures : Iterable of float
        The temperatures that each material is provided at
    nuclide_ndpp_lib : openmc.ndpp.NdppLibrary
        The nuclidic library to use for building the material

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
    num_delayed_groups : int
        Number of delayed neutron precursor groups
    chi : np.ndarray
        Data calculated by the pre-processor for later use in tallying of
        multi-group fission spectra data by OpenMC.
        The array has a shape of (len(chi_energy), num_groups,
        num_delayed_groups + 2) and the final index is ordered as follows:
        total, prompt, delayed[:num_delayed_groups]
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

    def __init__(self, material, temperatures, nuclide_ndpp_lib):
        self.name = material.name
        self.material = material
        self.kTs = np.array(temperatures) * K_BOLTZMANN
        self.nuclide_ndpp_lib = nuclide_ndpp_lib
        self.group_structure = nuclide_ndpp_lib.group_structure
        self.num_threads = nuclide_ndpp_lib.num_threads
        self.tolerance = nuclide_ndpp_lib.tolerance
        self.minimum_relative_threshold = \
            nuclide_ndpp_lib.minimum_relative_threshold
        self.scatter_format = nuclide_ndpp_lib.scatter_format
        self.order = nuclide_ndpp_lib.order
        self.freegas_cutoff = nuclide_ndpp_lib.freegas_cutoff
        self.freegas_method = nuclide_ndpp_lib.freegas_method
        self.elastic = {}
        self.elastic_energy = {}
        self.nu_inelastic = None
        self.inelastic = None
        self.inelastic_energy = None
        self.chi = {}
        self.chi_energy = {}
        self.fissionable = False

        # Get and store the microscopic NDPP objects, their atom densities
        # and the keys to use for the nearest temperature from the NDPP lib
        self.micro_ndpps = []
        self.atom_densities = []
        self.temp_keys = []

        nuclides = self.material.get_nuclide_atom_densities()

        # First find the nuclides which should include S(a,b) data
        # This will produce a list of microscopic S(a,b) NDPP objects and the
        # fraction of applicability for every nuclide in the problem; if a
        # nuclide has no S(a,b) data, then the list for that nuclide will be
        # empty.
        self.sabs = [[]] * len(nuclides)
        for i_sab, sab in enumerate(self.material._sab):
            # sab[0] is the library name, sab[1] is the fraction
            thermal_micro_ndpp = nuclide_ndpp_lib.get_by_name(sab[0])

            # Find nuclides for which the S(a,b) data is applicable
            sab_nucs = thermal_micro_ndpp.library.nuclides
            for n, nuclide in enumerate(nuclides):
                if nuclide in sab_nucs:
                    temp_keys = []
                    for temperature in temperatures:
                        kT = temperature * K_BOLTZMANN
                        # Find the nearest temperature index and store
                        iT = (np.abs(np.array(thermal_micro_ndpp.kTs) -
                                     kT)).argmin()
                        strT = \
                            "{}K".format(int(round(thermal_micro_ndpp.kTs[iT] /
                                                   K_BOLTZMANN)))
                        temp_keys.append(strT)

                    self.sabs[n].append((thermal_micro_ndpp, sab[1],
                                         temp_keys))

        # Now that we have the thermal nuclide info, go and get the standard
        # nuclidic info
        for nuclide in nuclides:
            # Save the atom density
            self.atom_densities.append(nuclides[nuclide][1])

            # Get & save the nuclidic NDPP libraries
            micro_ndpp = nuclide_ndpp_lib.get_by_name(nuclide)
            if micro_ndpp is None:
                raise ValueError("Nuclidic library does not contain " + nuclide)

            self.micro_ndpps.append(micro_ndpp)

            for temperature in temperatures:
                kT = temperature * K_BOLTZMANN
                # Find the nearest temperature index and store for later
                iT = (np.abs(np.array(micro_ndpp.kTs) - kT)).argmin()
                strT = \
                    "{}K".format(int(round(micro_ndpp.kTs[iT] / K_BOLTZMANN)))
                self.temp_keys.append(strT)

    @property
    def kTs(self):
        return self._kTs

    @kTs.setter
    def kTs(self, kTs):
        # This class needs its own setter since the Ndpp setter compares to the
        # library, which we do not have.
        if kTs is not None:
            cv.check_iterable_type('kTs', kTs, Real)
        self._kTs = kTs

    def process(self):
        """Combine the nuclidic data into material data
        """

        self._compute_inelastic()

        for ikT, kT in enumerate(self.kTs):
            # Get temperature string
            strT = "{}K".format(int(round(kT / K_BOLTZMANN)))
            # self._compute_elastic(kT, strT, self.temp_keys[ikT])
            self._compute_elastic(ikT, kT, strT, self.temp_keys[ikT])
            self._compute_chi(kT, strT, self.temp_keys[ikT])

    def _compute_inelastic(self):
        """Computes the pre-processed energy-angle data from the inelastic
        scattering reactions of a material. This method, unlike
        compute_elastic and compute_chi, computes the material-wise grid by
        taking a combined grid from all the nuclidic data sets and removing
        unnecessary points. This approach is used because all the relevant
        information is already contained within the nuclidic NDPP data and its'
        grid; for the other reaction the cross section grid also needs to be
        considered.
        """

        inelastic_energy = None
        inelastic = []
        nu_inelastic = []

        # Get the combined energy grids from the microscopic library
        for n in range(len(self.micro_ndpps)):
            micro_ndpp = self.micro_ndpps[n]

            # Now get and combine the grid data
            if micro_ndpp.inelastic_energy is not None:
                # The final inelastic energy point is going to be the one
                # NDPP included to keep a higher energy interpolation point;
                # including it here would result in the actual top energy point
                # being removed, so lets discard and re-add it later
                if inelastic_energy is None:
                    inelastic_energy = micro_ndpp.inelastic_energy[:-1]
                elif micro_ndpp.inelastic_energy is not None:
                    inelastic_energy = \
                        np.union1d(inelastic_energy,
                                   micro_ndpp.inelastic_energy[:-1])

        # Now we have the union grid we can go through and add in the
        # contributions from each isotope on to that grid
        if inelastic_energy is not None:
            energies_to_keep = np.ones(len(inelastic_energy), dtype=bool)
            energy_stencil = np.zeros(3)
            data_stencil = np.zeros((3, self.num_groups, self.num_angle))
            nu_data_stencil = np.zeros_like(data_stencil)
            for e, Ein in enumerate(inelastic_energy):
                # Create temporary storage for the data we are about to
                # calculate
                combined = np.zeros((self.num_groups, self.num_angle))
                nu_combined = np.zeros((self.num_groups,
                                        self.num_angle))
                for n in range(len(self.micro_ndpps)):
                    micro_ndpp = self.micro_ndpps[n]
                    if micro_ndpp.inelastic_energy is not None and \
                        Ein >= micro_ndpp.inelastic_energy[0]:

                        grid = micro_ndpp.inelastic_energy
                        data = micro_ndpp.inelastic
                        nu_data = micro_ndpp.nu_inelastic
                        # Find the corresponding point
                        if Ein < grid[-2]:
                            i = np.searchsorted(grid, Ein) - 1
                        else:
                            i = len(grid) - 2

                        # Get the interpolant
                        f = (Ein - grid[i]) / (grid[i + 1] - grid[i])

                        combined += self.atom_densities[n] * \
                            ((1. - f) * data[i] +
                             f * data[i + 1]).toarray()

                        nu_combined += self.atom_densities[n] * \
                            ((1. - f) * nu_data[i] +
                             f * nu_data[i + 1]).toarray()

                # Now start update our stencils to be used to figure out if the
                # central point (index 1) can be replaced with interpolation
                energy_stencil = np.roll(energy_stencil, -1, axis=0)
                energy_stencil[2] = Ein
                data_stencil = np.roll(data_stencil, -1, axis=0)
                data_stencil[2, ...] = combined[...]
                nu_data_stencil = np.roll(nu_data_stencil, -1, axis=0)
                nu_data_stencil[2, ...] = nu_combined[...]

                # Now, if we have a full stencil then we can figure
                # out if the middle information is necessary or not;
                # We do not perform this when e == len(...) - 1 since that
                # point was already covered by appending that point to
                # inelastic earlier
                if e > 1:
                    # find the interpolant to the middle energy point
                    f = (energy_stencil[1] - energy_stencil[0]) / \
                        (energy_stencil[2] - energy_stencil[0])

                    # And find what our interpolated values would be
                    interp = (1. - f) * data_stencil[0] + \
                        f * data_stencil[2]
                    nu_interp = (1. - f) * nu_data_stencil[0] + \
                        f * nu_data_stencil[2]

                    # Now see if the interpolable result is within our
                    # tolerance; if it is, keep it.
                    error = interp - data_stencil[1]
                    nu_error = nu_interp - nu_data_stencil[1]
                    # Avoid division by 0 errors since they are fully expected
                    # with our sparse results
                    with np.errstate(divide='ignore', invalid='ignore'):
                        error = np.abs(np.nan_to_num(
                            np.divide(error, data_stencil[1])))
                        nu_error = np.abs(np.nan_to_num(
                            np.divide(nu_error, nu_data_stencil[1])))

                    if np.any(error >= self.tolerance) or \
                        np.any(nu_error >= self.tolerance):
                        # Then we need the middle data point, so lets include
                        # it
                        inelastic.append(
                            SparseScatter(data_stencil[1, ...],
                                          self.minimum_relative_threshold,
                                          self.scatter_format))

                        nu_inelastic.append(
                            SparseScatter(nu_data_stencil[1, ...],
                                          self.minimum_relative_threshold,
                                          self.scatter_format))
                    else:
                        # Mark this energy grid as not being necessary
                        # Use the index 'e - 1' since the point we dont want
                        # is the previous one
                        energies_to_keep[e - 1] = False

                # Keep the start and end points since they can't be replaced
                # by interpolation on the fly
                if e == 0 or e == (len(inelastic_energy) - 1):
                    inelastic.append(
                        SparseScatter(combined,
                                      self.minimum_relative_threshold,
                                      self.scatter_format))

                    nu_inelastic.append(
                        SparseScatter(nu_combined,
                                      self.minimum_relative_threshold,
                                      self.scatter_format))

            # Remove energies we dont want
            inelastic_energy = inelastic_energy[energies_to_keep]

            # Finally, duplicate the top point to help with interpolation if
            # Ein is exactly equal in the Monte Carlo code to the top energy
            self.inelastic_energy = np.append(inelastic_energy,
                                              inelastic_energy[-1] + 1.e-1)
            self.nu_inelastic = np.append(nu_inelastic, [nu_inelastic[-1]])
            self.inelastic = np.append(inelastic, [inelastic[-1]])

            # And convert to a SparseScatters object
            self.inelastic = SparseScatters(self.inelastic)
            self.nu_inelastic = SparseScatters(self.nu_inelastic)

    def _compute_elastic(self, ikT, kT, strT, micro_ndpp_strT):
        """Computes the pre-processed energy-angle data from the elastic
        scattering reactions of a material. This method, similar to
        compute_chi, but unlike compute_inelastic, computes the material-wise
        grid by using adaptive grid generation. This method is used because
        the variability in the elastic cross section with energy is so large
        that a combined grid would effectively be a unionized grid of the
        constituent nuclide's cross section grids. This is likely overkill so
        instead an adaptive approach is used instead.
        """

        # Get the elastic cross section functions and the min/max Energies on
        # the grid
        xs_funcs = []
        sab_xs_funcs = []
        minE = np.finfo('float64').max
        maxE = np.finfo('float64').min
        for n in range(len(self.micro_ndpps)):
            if micro_ndpp_strT in self.micro_ndpps[n].library.reactions[2].xs:
                xs_funcs.append(
                    self.micro_ndpps[n].library.reactions[2].xs[
                        micro_ndpp_strT])
            if self.micro_ndpps[n].elastic_energy[strT][0] < minE:
                minE = self.micro_ndpps[n].elastic_energy[strT][0]

            # Get the maximum energy in the data; we will use index -2 since
            # the final index is just an added point for interpolation purposes
            if self.micro_ndpps[n].elastic_energy[strT][-2] > maxE:
                maxE = self.micro_ndpps[n].elastic_energy[strT][-2]

            if self.sabs[n]:
                sab_strT = self.sabs[n][0][2][ikT]
                try:
                    sab_elastic_xs = \
                        self.sabs[n][0][0].library.elastic_xs[sab_strT]
                except KeyError:
                    # Then there is no elastic data, return 0.
                    def sab_elastic_xs(Ein):
                        return 0.
                sab_inelastic_xs = \
                    self.sabs[n][0][0].library.inelastic_xs[sab_strT]
                sab_xs_funcs.append(lambda Ein: sab_elastic_xs(Ein) +
                                    sab_inelastic_xs(Ein))
            else:
                sab_xs_funcs.append(None)

        # Set the arguments for our linearize function, except dont yet include
        # the Ein grid points (the first argument), since that will be
        # dependent upon the thread's work
        func_args = (self.micro_ndpps, self.atom_densities, self.sabs,
                     (self.num_groups, self.num_angle), xs_funcs,
                     micro_ndpp_strT, sab_xs_funcs, ikT)
        linearize_args = (_linearizer_function_elastic, func_args,
                          self.tolerance, self.minimum_relative_threshold,
                          self.scatter_format)

        # Set the Ein grid
        Ein_grid = np.array([minE, maxE])

        inputs = [(Ein_grid[e: e + 2],) + linearize_args
                  for e in range(len(Ein_grid) - 1)]

        # Run in serial (Leaving architecture in place to eventually switch to
        # have a parallel capabilities)
        grid = [None] * len(inputs)
        results = [None] * len(inputs)
        for e, in_data in enumerate(inputs):
            grid[e], results[e] = linearizer_wrapper(in_data)

        # Now lets combine our grids together
        Ein_grid = np.concatenate(grid)
        results = np.concatenate(results)

        # Remove the non-unique entries obtained since the linearizer
        # includes the endpoints of each bracketed region
        Ein_grid, unique_indices = np.unique(Ein_grid, return_index=True)
        self.elastic[strT] = results[unique_indices]

        # Normalize results to remove the cross section dependence and
        # any numerical errors
        for e in range(len(Ein_grid)):
            if self.scatter_format == 'legendre':
                divisor = np.sum(self.elastic[strT][e][:, 0])
            else:
                divisor = np.sum(self.elastic[strT][e][:, :])
            if divisor > 0.:
                self.elastic[strT][e] = self.elastic[strT][e] / divisor

        # Finally add a top point to use as interpolation
        Ein_grid = np.append(Ein_grid, Ein_grid[-1] + 1.e-1)
        self.elastic_energy[strT] = Ein_grid
        self.elastic[strT] = np.append(self.elastic[strT],
                                       [self.elastic[strT][-1]])
        self.elastic[strT] = SparseScatters(self.elastic[strT])

    def _compute_chi(self, kT, strT, micro_ndpp_strT):
        """Computes the pre-processed energy-angle data from the fission
        spectra reactions of a material. This method, similar to
        compute_elastic, but unlike compute_inelastic, computes the
        material-wise grid by using adaptive grid generation. This method is
        used because the variability in the fission cross section with energy
        is so large that a combined grid would effectively be a unionized grid
        of the constituent nuclide's cross section grids. This is likely
        overkill so instead an adaptive approach is used instead.
        """

        # Get the fission cross section functions and the min/max energies on
        # the grid
        xs_funcs = []
        minE = np.finfo('float64').max
        maxE = np.finfo('float64').min
        for n in range(len(self.micro_ndpps)):
            if self.micro_ndpps[n].fissionable:
                if micro_ndpp_strT in self.micro_ndpps[n].library.reactions[18].xs:
                    xs_funcs.append(
                        self.micro_ndpps[n].library.reactions[18].xs[
                            micro_ndpp_strT])
                if self.micro_ndpps[n].chi_energy[strT][0] < minE:
                    minE = self.micro_ndpps[n].chi_energy[strT][0]

                # Get the maximum energy in the data; we will use index -2
                # since the final index is just an added point for
                # interpolation purposes
                if self.micro_ndpps[n].chi_energy[strT][-2] > maxE:
                    maxE = self.micro_ndpps[n].chi_energy[strT][-2]

                self.fissionable = True
                self.num_delayed_groups = \
                    self.micro_ndpps[n].num_delayed_groups

        if not self.fissionable:
            self.num_delayed_groups = 0
            return

        # Set the arguments for our linearize function, except dont yet include
        # the Ein grid points (the first argument), since that will be
        # dependent upon the thread's work
        func_args = (self.micro_ndpps, self.atom_densities,
                     (self.num_groups, self.num_delayed_groups + 2), xs_funcs,
                     micro_ndpp_strT)
        linearize_args = (_linearizer_function_chi, func_args,
                          self.tolerance, self.minimum_relative_threshold,
                          self.scatter_format)

        # Set the Ein grid
        Ein_grid = np.array([minE, maxE])

        inputs = [(Ein_grid[e: e + 2],) + linearize_args
                  for e in range(len(Ein_grid) - 1)]

        # Run in serial (Leaving architecture in place to eventually switch to
        # have a parallel capabilities)
        grid = [None] * len(inputs)
        results = [None] * len(inputs)

        for e, in_data in enumerate(inputs):
            grid[e], results[e] = linearizer_wrapper(in_data)

        # Now lets combine our grids together
        Ein_grid = np.concatenate(grid)
        results = np.concatenate(results)

        # Remove the non-unique entries obtained since the linearizer
        # includes the endpoints of each bracketed region
        Ein_grid, unique_indices = np.unique(Ein_grid, return_index=True)
        results = results[unique_indices]

        # With chi data, we only care about the normalized secondary energy
        # data, so lets normalize
        for e in range(len(Ein_grid)):
            for tpd in range(2 + self.num_delayed_groups):
                results[e].data[:, tpd] /= np.sum(results[e].data[:, tpd])

        # Add a top point to use as interpolation
        Ein_grid = np.append(Ein_grid, Ein_grid[-1] + 1.e-1)
        results = np.append(results, [results[-1]])

        # Self was initialized as not being fissionable, if it actually is,
        # make note now and store the results
        if self.fissionable:
            self.chi_energy[strT] = Ein_grid[:]
            self.chi[strT] = SparseScatters(results[:])


def _linearizer_function_elastic(Ein, micro_ndpps, atom_densities, sabs,
                                 array_shape, xs_funcs, strT, sab_xs_funcs,
                                 ikT):
    combined = np.zeros(array_shape)
    for n in range(len(micro_ndpps)):
        # First check/get the thermal data
        if sabs[n]:
            # Then the list is populated and there is info to apply for this
            # nuclide; so now check the energy bounds
            thermal_micro_ndpp = sabs[n][0][0]
            sab_strT = sabs[n][0][2][ikT]
            sab_fraction = sabs[n][0][1]
            # Remember compare to the point at index -2 because -1 was added
            # to help with downstream interpolation and is not physical
            if Ein <= thermal_micro_ndpp.elastic_energy[sab_strT][-2]:
                grid = thermal_micro_ndpp.elastic_energy[sab_strT]
                data = thermal_micro_ndpp.elastic[sab_strT]
                macro_xs = \
                    sab_xs_funcs[n](Ein) * atom_densities[n] * sab_fraction

                # Find the corresponding grid point
                if Ein <= grid[0]:
                    i = 0
                elif Ein < grid[-2]:
                    i = np.searchsorted(grid, Ein) - 1
                else:
                    i = len(grid) - 2

                # Get the interpolant
                f = (Ein - grid[i]) / (grid[i + 1] - grid[i])

                combined += \
                    macro_xs * ((1. - f) * data[i] + f * data[i + 1]).toarray()

                # Set the sab_fraction so the nuclide info is applied as needed
                sab_fraction = sabs[n][0][1]
            else:
                # Then we are above the range, set the fraction to 1 so the
                # nuclidic info is applied in full.
                sab_fraction = 0.
        else:
            # Then we have no S(a,b) data for this nuclide, set the fraction
            # to 1 so the nuclidic info is applied in full.
            sab_fraction = 0.

        if sab_fraction < 1.:
            grid = micro_ndpps[n].elastic_energy[strT]
            data = micro_ndpps[n].elastic[strT]
            macro_xs = \
                xs_funcs[n](Ein) * atom_densities[n] * (1. - sab_fraction)

            # Find the corresponding grid point
            if Ein <= grid[0]:
                i = 0
            elif Ein < grid[-2]:
                i = np.searchsorted(grid, Ein) - 1
            else:
                i = len(grid) - 2

            # Get the interpolant
            f = (Ein - grid[i]) / (grid[i + 1] - grid[i])

            combined += \
                macro_xs * ((1. - f) * data[i] + f * data[i + 1]).toarray()

    return combined


def _linearizer_function_chi(Ein, micro_ndpps, atom_densities, array_shape,
                             xs_funcs, strT):
    combined = np.zeros(array_shape)
    for n in range(len(micro_ndpps)):
        if micro_ndpps[n].fissionable:
            grid = micro_ndpps[n].chi_energy[strT]
            data = micro_ndpps[n].chi[strT]
            macro_xs = xs_funcs[n](Ein) * atom_densities[n]

            # Find the corresponding grid point
            if Ein <= grid[0]:
                i = 0
            elif Ein < grid[-2]:
                i = np.searchsorted(grid, Ein) - 1
            else:
                i = len(grid) - 2

            # Get the interpolant
            f = (Ein - grid[i]) / (grid[i + 1] - grid[i])

            combined += \
                macro_xs * ((1. - f) * data[i] + f * data[i + 1]).toarray()

    return combined
