from collections import Iterable
from numbers import Real, Integral
from warnings import warn

import numpy as np
import scipy.integrate as sint
import scipy.special as ss

import openmc.checkvalue as cv
from openmc.stats import Tabular, Univariate, Discrete, Mixture
from .function import Tabulated1D, INTERPOLATION_SCHEME
from .angle_energy import AngleEnergy
from .data import EV_PER_MEV
from .endf import get_list_record, get_tab2_record


class KalbachMann(AngleEnergy):
    """Kalbach-Mann distribution

    Parameters
    ----------
    breakpoints : Iterable of int
        Breakpoints defining interpolation regions
    interpolation : Iterable of int
        Interpolation codes
    energy : Iterable of float
        Incoming energies at which distributions exist
    energy_out : Iterable of openmc.stats.Univariate
        Distribution of outgoing energies corresponding to each incoming energy
    precompound : Iterable of openmc.data.Tabulated1D
        Precompound factor 'r' as a function of outgoing energy for each
        incoming energy
    slope : Iterable of openmc.data.Tabulated1D
        Kalbach-Chadwick angular distribution slope value 'a' as a function of
        outgoing energy for each incoming energy

    Attributes
    ----------
    breakpoints : Iterable of int
        Breakpoints defining interpolation regions
    interpolation : Iterable of int
        Interpolation codes
    energy : Iterable of float
        Incoming energies at which distributions exist
    energy_out : Iterable of openmc.stats.Univariate
        Distribution of outgoing energies corresponding to each incoming energy
    precompound : Iterable of openmc.data.Tabulated1D
        Precompound factor 'r' as a function of outgoing energy for each
        incoming energy
    slope : Iterable of openmc.data.Tabulated1D
        Kalbach-Chadwick angular distribution slope value 'a' as a function of
        outgoing energy for each incoming energy

    """

    def __init__(self, breakpoints, interpolation, energy, energy_out,
                 precompound, slope):
        super(KalbachMann, self).__init__()
        self.breakpoints = breakpoints
        self.interpolation = interpolation
        self.energy = energy
        self.energy_out = energy_out
        self.precompound = precompound
        self.slope = slope

    @property
    def breakpoints(self):
        return self._breakpoints

    @property
    def interpolation(self):
        return self._interpolation

    @property
    def energy(self):
        return self._energy

    @property
    def energy_out(self):
        return self._energy_out

    @property
    def precompound(self):
        return self._precompound

    @property
    def slope(self):
        return self._slope

    @breakpoints.setter
    def breakpoints(self, breakpoints):
        cv.check_type('Kalbach-Mann breakpoints', breakpoints,
                      Iterable, Integral)
        self._breakpoints = breakpoints

    @interpolation.setter
    def interpolation(self, interpolation):
        cv.check_type('Kalbach-Mann interpolation', interpolation,
                      Iterable, Integral)
        self._interpolation = interpolation

    @energy.setter
    def energy(self, energy):
        cv.check_type('Kalbach-Mann incoming energy', energy,
                      Iterable, Real)
        self._energy = energy

    @energy_out.setter
    def energy_out(self, energy_out):
        cv.check_type('Kalbach-Mann distributions', energy_out,
                      Iterable, Univariate)
        self._energy_out = energy_out

    @precompound.setter
    def precompound(self, precompound):
        cv.check_type('Kalbach-Mann precompound factor', precompound,
                      Iterable, Tabulated1D)
        self._precompound = precompound

    @slope.setter
    def slope(self, slope):
        cv.check_type('Kalbach-Mann slope', slope, Iterable, Tabulated1D)
        self._slope = slope

    def sample(self, Ein, cm, awr):
        # Generate the incoming distribution via interpolation if needed
        if Ein <= self._energy[0]:
            i = 0
            r = 0.
        elif Ein > self._energy[-1]:
            i = len(self._energy) - 2
            r = 1.
        else:
            i = np.searchsorted(self._energy, Ein) - 1
            r = (Ein - self._energy[i]) / \
                (self._energy[i + 1] - self._energy[i])

        if r > np.random.random():
            l = i + 1
        else:
            l = i

        E_i_1 = self._energy_out[i].x[0]
        E_i_K = self._energy_out[i].x[-1]
        E_i1_1 = self._energy_out[i + 1].x[0]
        E_i1_K = self._energy_out[i + 1].x[-1]

        E_1 = E_i_1 + r * (E_i1_1 - E_i_1)
        E_K = E_i_K + r * (E_i1_K - E_i_K)

        # Determine outgoing energy bin
        r1 = np.random.random()
        c_k = self._energy_out[l].c[0]
        for k in range(len(self._energy_out[l]) - 1):
            c_k1 = self._energy_out[l].c[k + 1]
            if r1 < c_k1:
                break
            c_k = c_k1

        # Check to make sure k is <= NP - 1
        k = min(k, len(self._energy_out[l].x) - 2)

        E_l_k = self._energy_out[l].x[k]
        p_l_k = self._energy_out[l].p[k]

        # import pdb; pdb.set_trace()
        interp_index = 0
        interp = self._interpolation[interp_index]

        if interp == 1:
            # Histogram
            if p_l_k > 0.:
                E_out = E_l_k + (r1 - c_k) / p_l_k
            else:
                E_out = E_l_k
            km_r = self._precompound[l].y[k]
            km_a = self._slope[l].y[k]
        elif interp == 2:
            # linear-linear
            E_l_k1 = self._energy_out[l].x[k + 1]
            p_l_k1 = self._energy_out[l].p[k + 1]

            frac = (p_l_k1 - p_l_k) / (E_l_k1 - E_l_k)
            if frac == 0.:
                E_out = E_l_k + (r1 - c_k) / p_l_k
            else:
                E_out = E_l_k + (np.sqrt(max(0., p_l_k * p_l_k +
                                             2. * frac * (r1 - c_k))) -
                                 p_l_k) / frac

            km_r = self._precompound[l].y[k] + (E_out - E_l_k) / \
                (E_l_k1 - E_l_k) * \
                (self._precompound[l].y[k + 1] - self._precompound[l].y[k])
            km_a = self._slope[l].y[k] + (E_out - E_l_k) / \
                (E_l_k1 - E_l_k) * \
                (self._slope[l].y[k + 1] - self._slope[l].y[k])

        if l == i:
            E_out = E_1 + (E_out - E_i_1) * (E_K - E_1) / (E_i_K - E_i_1)
        else:
            E_out = E_1 + (E_out - E_i1_1) * \
                (E_K - E_1) / (E_i1_K - E_i1_1)

        # Sampled correlation angle from KM parameters
        if np.random.random() > km_r:
            T = (2. * np.random.random() - 1.) * np.sinh(km_a)
            mu = np.log(T + np.sqrt(T * T + 1.)) / km_a
        else:
            r1 = np.random.random()
            mu = np.log(r1 * np.exp(km_a) + (1. - r1) * np.exp(-km_a)) / km_a


        # if scattering system is in center-of-mass, transfer cosine of
        # scattering angle and outgoing energy from CM to LAB
        if cm:
            E_cm = E_out

            # determine outgoing energy in lab
            if Ein * E_cm < 0.:
                import pdb; pdb.set_trace()
            E_out = E_cm + (Ein + 2. * mu * (awr + 1.) *
                            np.sqrt(Ein * E_cm)) / ((awr + 1.) * (awr + 1.))
            # determine outgoing angle in lab
            mu = mu * np.sqrt(E_cm / E_out) + \
                1. / (awr + 1.) * np.sqrt(Ein / E_out)

        # Because of floating-point roundoff, it may be possible for mu to be
        # outside of the range [-1,1). In these cases, we just set mu to
        # exactly -1 or 1
        if mu > 1.:
            mu = 1.
        elif mu < -1.:
            mu = -1.

        return E_out, mu

    def to_hdf5(self, group):
        """Write distribution to an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to

        """
        group.attrs['type'] = np.string_('kalbach-mann')

        dset = group.create_dataset('energy', data=self.energy)
        dset.attrs['interpolation'] = np.vstack((self.breakpoints,
                                                 self.interpolation))

        # Determine total number of (E,p,r,a) tuples and create array
        n_tuple = sum(len(d) for d in self.energy_out)
        distribution = np.empty((5, n_tuple))

        # Create array for offsets
        offsets = np.empty(len(self.energy_out), dtype=int)
        interpolation = np.empty(len(self.energy_out), dtype=int)
        n_discrete_lines = np.empty(len(self.energy_out), dtype=int)
        j = 0

        # Populate offsets and distribution array
        for i, (eout, km_r, km_a) in enumerate(zip(
                self.energy_out, self.precompound, self.slope)):
            n = len(eout)
            offsets[i] = j

            if isinstance(eout, Mixture):
                discrete, continuous = eout.distribution
                n_discrete_lines[i] = m = len(discrete)
                interpolation[i] = 1 if continuous.interpolation == 'histogram' else 2
                distribution[0, j:j+m] = discrete.x
                distribution[1, j:j+m] = discrete.p
                distribution[2, j:j+m] = discrete.c
                distribution[0, j+m:j+n] = continuous.x
                distribution[1, j+m:j+n] = continuous.p
                distribution[2, j+m:j+n] = continuous.c
            else:
                if isinstance(eout, Tabular):
                    n_discrete_lines[i] = 0
                    interpolation[i] = 1 if eout.interpolation == 'histogram' else 2
                elif isinstance(eout, Discrete):
                    n_discrete_lines[i] = n
                    interpolation[i] = 1
                distribution[0, j:j+n] = eout.x
                distribution[1, j:j+n] = eout.p
                distribution[2, j:j+n] = eout.c

            distribution[3, j:j+n] = km_r.y
            distribution[4, j:j+n] = km_a.y
            j += n

        # Create dataset for distributions
        dset = group.create_dataset('distribution', data=distribution)

        # Write interpolation as attribute
        dset.attrs['offsets'] = offsets
        dset.attrs['interpolation'] = interpolation
        dset.attrs['n_discrete_lines'] = n_discrete_lines

    @classmethod
    def from_hdf5(cls, group):
        """Generate Kalbach-Mann distribution from HDF5 data

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to read from

        Returns
        -------
        openmc.data.KalbachMann
            Kalbach-Mann energy distribution

        """
        interp_data = group['energy'].attrs['interpolation']
        energy_breakpoints = interp_data[0, :]
        energy_interpolation = interp_data[1, :]
        energy = group['energy'].value

        data = group['distribution']
        offsets = data.attrs['offsets']
        interpolation = data.attrs['interpolation']
        n_discrete_lines = data.attrs['n_discrete_lines']

        energy_out = []
        precompound = []
        slope = []
        n_energy = len(energy)
        for i in range(n_energy):
            # Determine length of outgoing energy distribution and number of
            # discrete lines
            j = offsets[i]
            if i < n_energy - 1:
                n = offsets[i+1] - j
            else:
                n = data.shape[1] - j
            m = n_discrete_lines[i]

            # Create discrete distribution if lines are present
            if m > 0:
                eout_discrete = Discrete(data[0, j:j+m], data[1, j:j+m])
                eout_discrete.c = data[2, j:j+m]
                p_discrete = eout_discrete.c[-1]

            # Create continuous distribution
            if m < n:
                interp = INTERPOLATION_SCHEME[interpolation[i]]
                eout_continuous = Tabular(data[0, j+m:j+n], data[1, j+m:j+n], interp)
                eout_continuous.c = data[2, j+m:j+n]

            # If both continuous and discrete are present, create a mixture
            # distribution
            if m == 0:
                eout_i = eout_continuous
            elif m == n:
                eout_i = eout_discrete
            else:
                eout_i = Mixture([p_discrete, 1. - p_discrete],
                                 [eout_discrete, eout_continuous])

            km_r = Tabulated1D(data[0, j:j+n], data[3, j:j+n])
            km_a = Tabulated1D(data[0, j:j+n], data[4, j:j+n])

            energy_out.append(eout_i)
            precompound.append(km_r)
            slope.append(km_a)

        return cls(energy_breakpoints, energy_interpolation,
                   energy, energy_out, precompound, slope)

    @classmethod
    def from_ace(cls, ace, idx, ldis):
        """Generate Kalbach-Mann energy-angle distribution from ACE data

        Parameters
        ----------
        ace : openmc.data.ace.Table
            ACE table to read from
        idx : int
            Index in XSS array of the start of the energy distribution data
            (LDIS + LOCC - 1)
        ldis : int
            Index in XSS array of the start of the energy distribution block
            (e.g. JXS[11])

        Returns
        -------
        openmc.data.KalbachMann
            Kalbach-Mann energy-angle distribution

        """
        # Read number of interpolation regions and incoming energies
        n_regions = int(ace.xss[idx])
        n_energy_in = int(ace.xss[idx + 1 + 2*n_regions])

        # Get interpolation information
        idx += 1
        if n_regions > 0:
            breakpoints = ace.xss[idx:idx + n_regions].astype(int)
            interpolation = ace.xss[idx + n_regions:idx + 2*n_regions].astype(int)
        else:
            breakpoints = np.array([n_energy_in])
            interpolation = np.array([2])

        # Incoming energies at which distributions exist
        idx += 2*n_regions + 1
        energy = ace.xss[idx:idx + n_energy_in]*EV_PER_MEV

        # Location of distributions
        idx += n_energy_in
        loc_dist = ace.xss[idx:idx + n_energy_in].astype(int)

        # Initialize variables
        energy_out = []
        km_r = []
        km_a = []

        # Read each outgoing energy distribution
        for i in range(n_energy_in):
            idx = ldis + loc_dist[i] - 1

            # intt = interpolation scheme (1=hist, 2=lin-lin)
            INTTp = int(ace.xss[idx])
            intt = INTTp % 10
            n_discrete_lines = (INTTp - intt)//10
            if intt not in (1, 2):
                warn("Interpolation scheme for continuous tabular distribution "
                     "is not histogram or linear-linear.")
                intt = 2

            n_energy_out = int(ace.xss[idx + 1])
            data = ace.xss[idx + 2:idx + 2 + 5*n_energy_out].copy()
            data.shape = (5, n_energy_out)
            data[0,:] *= EV_PER_MEV

            # Create continuous distribution
            eout_continuous = Tabular(data[0][n_discrete_lines:],
                                      data[1][n_discrete_lines:]/EV_PER_MEV,
                                      INTERPOLATION_SCHEME[intt],
                                      ignore_negative=True)
            eout_continuous.c = data[2][n_discrete_lines:]
            if np.any(data[1][n_discrete_lines:] < 0.0):
                warn("Kalbach-Mann energy distribution has negative "
                     "probabilities.")

            # If discrete lines are present, create a mixture distribution
            if n_discrete_lines > 0:
                eout_discrete = Discrete(data[0][:n_discrete_lines],
                                         data[1][:n_discrete_lines])
                eout_discrete.c = data[2][:n_discrete_lines]
                if n_discrete_lines == n_energy_out:
                    eout_i = eout_discrete
                else:
                    p_discrete = min(sum(eout_discrete.p), 1.0)
                    eout_i = Mixture([p_discrete, 1. - p_discrete],
                                     [eout_discrete, eout_continuous])
            else:
                eout_i = eout_continuous

            energy_out.append(eout_i)
            km_r.append(Tabulated1D(data[0], data[3]))
            km_a.append(Tabulated1D(data[0], data[4]))

        return cls(breakpoints, interpolation, energy, energy_out, km_r, km_a)

    @classmethod
    def from_endf(cls, file_obj):
        """Generate Kalbach-Mann distribution from an ENDF evaluation

        Parameters
        ----------
        file_obj : file-like object
            ENDF file positioned at the start of the Kalbach-Mann distribution

        Returns
        -------
        openmc.data.KalbachMann
            Kalbach-Mann energy-angle distribution

        """
        params, tab2 = get_tab2_record(file_obj)
        lep = params[3]
        ne = params[5]
        energy = np.zeros(ne)
        n_discrete_energies = np.zeros(ne, dtype=int)
        energy_out = []
        precompound = []
        slope = []
        for i in range(ne):
            items, values = get_list_record(file_obj)
            energy[i] = items[1]
            n_discrete_energies[i] = items[2]
            # TODO: split out discrete energies
            n_angle = items[3]
            n_energy_out = items[5]
            values = np.asarray(values)
            values.shape = (n_energy_out, n_angle + 2)

            # Outgoing energy distribution at the i-th incoming energy
            eout_i = values[:,0]
            eout_p_i = values[:,1]
            energy_out_i = Tabular(eout_i, eout_p_i, INTERPOLATION_SCHEME[lep])
            energy_out.append(energy_out_i)

            # Precompound and slope factors for Kalbach-Mann
            r_i = values[:,2]
            if n_angle == 2:
                a_i = values[:,3]
            else:
                a_i = np.zeros_like(r_i)
            precompound.append(Tabulated1D(eout_i, r_i))
            slope.append(Tabulated1D(eout_i, a_i))

        return cls(tab2.breakpoints, tab2.interpolation, energy,
                   energy_out, precompound, slope)

    @classmethod
    def example(cls):
        """ Builds an example Kalbach-Mann object for testing

        """

        from openmc.stats.univariate import Tabular
        xlo = np.array([0., 14110.19])
        xhi = np.array([0., 14110.19, 28220.38, 47033.96, 65847.55, 84661.13,
                        103474.7, 122288.3, 141101.9, 159915.5, 178729.1,
                        206949.4, 244576.6, 282203.8, 319830.9, 357458.1,
                        470339.6])
        breakpoints = [2]
        interpolation = [2]
        energy = np.array([1.66516E7, 1.7E7])
        energy_out = [Tabular(xlo, [7.087077E-5, 0.]),
                      Tabular(xhi,
            np.array([8.34974100e-07, 1.51966200e-06, 2.03220800e-06,
                      2.49257100e-06, 2.87962900e-06, 3.22027400e-06,
                      3.52808000e-06, 3.81105200e-06, 4.07438700e-06,
                      4.32168700e-06, 4.38502700e-06, 3.64483100e-06,
                      2.73362300e-06, 1.82241500e-06, 9.11207600e-07,
                      3.75873100e-08, 0.00000000e+00]))]
        energy_out[0].c = np.array([0., 1.])
        energy_out[1].c = \
            np.array([0., 0.01178164, 0.03322436, 0.07145747, 0.11835168,
                      0.17252782, 0.23311267, 0.29948856, 0.37118817,
                      0.44784206, 0.52914855, 0.65289533, 0.79004012,
                      0.8928987, 0.9614709, 0.99575709, 1.])
        precompound = [Tabulated1D(xlo, np.array([0., 0.])), Tabulated1D(xhi,
                np.array([0.4217524, 0.4006648, 0.3795772, 0.3514603,
                          0.3233435, 0.2952267, 0.2671099, 0.238993, 0.2108762,
                          0.1827594, 0.1546425, 0.1124673, 0.05623365, 0., 0.,
                          0., 0.]))]

        slope = []
        slope.append(Tabulated1D(xlo, np.array([0.4054679, 0.4060959])))
        slope.append(Tabulated1D(xhi,
            np.array([0.4054679, 0.4060959, 0.406724, 0.4075616, 0.4083995,
                      0.4092375, 0.4100758, 0.4109142, 0.4117529, 0.4125918,
                      0.4134309, 0.41469, 0.4163694, 0.4180498, 0.419731,
                      0.421413, 0.4264644])))

        return cls(breakpoints, interpolation, energy, energy_out, precompound,
                   slope)
