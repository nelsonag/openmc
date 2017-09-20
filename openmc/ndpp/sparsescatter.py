from __future__ import division
import operator
from copy import deepcopy
from numbers import Real

import numpy as np


def _generic_compare(self, other, operator):
    """Implement <, <=, == , !=, >, >= for the SparseScatter Objects."""

    if not isinstance(other, Real):
        raise NotImplementedError('Comparing to a non-Real number is not '
                                  'supported')

    return operator(self.data, other)


def _generic_addsub(self, other, operator):
    """The +. - operations have the same code except for the actual
    performance of the operation. This function therefore aggregates this
    code and generalizes the operation for the purposes of reducing code
    duplication"""

    if isinstance(other, Real):
        raise NotImplementedError('adding or subtracting a scalar to a '
                                  'sparse matrix is not supported')

    elif isinstance(other, np.ndarray):
        if self.shape != other.shape:
            raise ValueError("Shapes must be identical")
        else:
            sother = SparseScatter(other)
            return operator(self, sother)

    elif isinstance(other, SparseScatter):
        if self.shape != other.shape:
            raise ValueError("Shapes must be identical")
        elif self.gout_min == -1:
            # Then self is just 0s, so just pass back other
            return deepcopy(other)
        elif other.gout_min == -1:
            # Then other is just 0s, so just pass back self
            return deepcopy(self)
        else:
            result = SparseScatter()
            if self.gout_min == other.gout_min and \
                self.gout_max == other.gout_max:

                # Then the data just needs to be combined
                result.gout_min = self.gout_min
                result.gout_max = self.gout_max
                result.shape = self.shape
                result.data = operator(self.data, other.data)
            else:
                # Then we need to operate only in the ranges of interest
                # First find the extend and size for the data
                gout_min = min(self.gout_min, other.gout_min)
                gout_max = max(self.gout_max, other.gout_max)
                data = np.zeros((gout_max - gout_min + 1, self.shape[1]))
                self_start = self.gout_min - gout_min
                self_end = self.gout_max - self.gout_min + self_start
                other_start = other.gout_min - gout_min
                other_end = other.gout_max - other.gout_min + other_start

                data[self_start: self_end + 1] = self.data

                data[other_start: other_end + 1] = \
                    operator(data[other_start: other_end + 1], other.data)

                result.gout_min = gout_min
                result.gout_max = gout_max
                result.shape = self.shape
                result.data = data
            return result

    else:
        return NotImplemented


def _generic_muldiv(self, other, operator):
    """The * and \ operations have the same code except for the actual
    performance of the operation. This function therefore aggregates this
    code and generalizes the operation for the purposes of reducing code
    duplication"""

    if isinstance(other, Real):
        result = deepcopy(self)
        # The shape doesn't change, so we can just operate on the data
        result.data = operator(result.data, other)
        return result

    elif isinstance(other, np.ndarray):
        if self.shape != other.shape:
            raise ValueError("Shapes must be identical")
        else:
            sother = SparseScatter(other)
            return operator(self, sother)

    elif isinstance(other, SparseScatter):
        if self.shape != other.shape:
            raise ValueError("Shapes must be identical")
        elif self.gout_min == -1:
            # Then self is just 0s, so the combination will be too;
            # since self is 0s, we can just pass that back
            return deepcopy(self)
        elif other.gout_min == -1:
            if operator == operator.mul:
                # Then other is just 0s, so the combination will be too;
                # since other is 0s, we can just pass that back
                return deepcopy(other)
            else:
                # Then we will be dividing by 0; just return self and set the
                # values of self.data to NaN
                result = deepcopy(self)
                result.data = np.NaN
                return result
        else:
            if self.gout_min == other.gout_min and \
                self.gout_max == other.gout_max:

                # Then the data just needs to be combined
                result = SparseScatter()
                result.gout_min = self.gout_min
                result.gout_max = self.gout_max
                result.shape = self.shape
                result.data = operator(self.data, other.data)
            else:
                # Given the rarity of an element-by-element matrix mult/div,
                # lets just handle the remaining case by converting to ndarrays
                # and reconverting back to sparse data
                result = SparseScatter(operator(self.toarray(),
                                                other.toarray()))
            return result

    else:
        return NotImplemented


class SparseScatters(object):
    """This class is a container for an array of SparseScatter objects.

    Parameters
    ----------
    scatters : Iterable of SparseScatter objects
        The array of scattering objects to work with.

    Attributes
    ----------
    scatters : np.ndarray of SparseScatter
        The array of scattering objects to work with.

    """

    def __init__(self, scatters):
        self.scatters = np.array(scatters, dtype=SparseScatter)

    def __getitem__(self, indices):
        if isinstance(indices, tuple):
            if len(indices) == 3:
                # Get the indices
                e, g, L = indices

                if isinstance(e, slice):
                    e_start, e_stop, e_step = e.indices(len(self.scatters))
                    e_size = (e_stop - e_start) // e_step

                    result = [None] * e_size
                    for ie in range(e_start, e_stop, e_step):
                        result[ie] = self.scatters[ie][g, L]
                    return np.array(result)
                else:
                    # Get the entry in our array
                    my_scatter = self.scatters[e]
                    # And pass on the remaining indices to SparseScatter's
                    # getitem
                    return my_scatter[g, L]
            elif len(indices) == 1:
                # Then we will assume this is to get access to the
                # SparseScatter object
                return self.scatters[indices[0]]
        elif isinstance(indices, int):
            # Then we will assume this is to get access to the SparseScatter
            # object
            return self.scatters[indices]
        else:
            raise ValueError("Invalid Indices")

    def __setitem__(self, indices, value):
        if isinstance(indices, tuple):
            if len(indices) == 3:
                # Get the indices
                e, g, L = indices

                if isinstance(e, slice):
                    e_start, e_stop, e_step = e.indices(len(self.scatters))

                    for ie in range(e_start, e_stop, e_step):
                        self.scatters[ie][g, L] = value[ie, g, L]
                else:
                    # Set the entry in our array
                    self.scatters[e][g, L] = value
            elif len(indices) == 1:
                self.scatters[indices[0]] = value
        elif isinstance(indices, int):
            # Then we will assume this is to get access to the SparseScatter
            # object
            return self.scatters[indices]
        else:
            raise ValueError("Invalid Indices")

    def __len__(self):
        return self.scatters.shape[0]

    def append(self, value):
        if isinstance(value, SparseScatter):
            self.scatters = np.append(self.scatters, [value], axis=0)
        else:
            raise ValueError("Append can only be used to add SparseScatter "
                             "objects")


class SparseScatter(object):
    """This class is a sparse matrix representation specialized to the needs
    of NDPP. A sparse matrix representation is needed because outgoing
    scattering and fission matrices can become quite sparse, for example, when
    for heavy isotopes which do not have significant down-scatter.
    However, the standard sparse representations are non-optimum since the
    the sparsity only exists on the outgoing group dimension, not the dimension
    representating the Legendre coefficients or change-in-angle bins.

    Parameters
    ----------
    matrix : np.ndarray
        Dense matrix to convert to a sparse matrix
    minimum_relative_threshold : float, optional
        Values with an absolute value less than this threshold will be treated
        as a zero.  This is done on a relative basis, relative to the sum of
        the P0 moments (if scatter_format is 'legendre', or the sum of the
        whole matrix (if scatter_format is 'histogram').
    scatter_format : {'legendre', 'histogram'}, optional
        Mathematical scatter_format of the scattering matrices: either a
        Legendre expansion of the order defined in the order parameter, or
        a histogram scatter_format with a number of bins defined by the order
        parameter.

    Attributes
    ----------
    shape : 2-tuple
        Shape of the matrix
    data : np.ndarray
        Flattened array containing only the relevant data, the first dimension
        is the flattened outgoing group index and the second is the angular
        representation, with the same size as the source matrix.
    gout_min : int
        Lowest outgoing group index
    gout_max : int
        Highest outgoing group index

    """

    def __init__(self, matrix=None, minimum_relative_threshold=None,
                 scatter_format=None):
        if isinstance(matrix, np.ndarray):

            # Apply the minimum threshold
            if minimum_relative_threshold is not None:
                if scatter_format == 'legendre':
                    pre_thin_sum = np.sum(matrix[:, 0])
                    threshold = pre_thin_sum * minimum_relative_threshold

                    for gout in range(matrix.shape[0]):
                        if matrix[gout, 0] < threshold:
                            matrix[gout, :] = 0.
                else:
                    pre_thin_sum = np.sum(matrix)
                    threshold = pre_thin_sum * minimum_relative_threshold
                    for gout in range(matrix.shape[0]):
                        if np.sum(matrix[gout, :]) < threshold:
                            matrix[gout, :] = 0.

            # Find the nonzero entries using numpy
            if scatter_format == 'legendre':
                nz = np.nonzero(matrix[:, 0])
            else:
                nz = np.nonzero(np.sum(matrix, axis=-1))

            # Now get our outgoing group bounds
            if len(nz[0]) == 0:
                # It is possible that the zero is all zeros, if so, set the
                # gout_min/max to -1
                self.gout_min = -1
                self.gout_max = -1
            else:
                # If nz has values, then the top and bottom extents are simply
                # the first and last entries of our nz list
                self.gout_min = nz[0][0]
                self.gout_max = nz[0][-1]

            # Now gather the relevant data
            self.data = matrix[self.gout_min: self.gout_max + 1, :]
            self.shape = matrix.shape

        elif matrix is None:
            # Then we just want a blank initialization
            self.data = None
            self.shape = (0, 0)
            self.gout_min = 0
            self.gout_max = 0

    def __getitem__(self, indices):
        if isinstance(indices, tuple) and len(indices) == 2:
            # Get the indices
            g, L = indices

            g_is_slice = isinstance(g, slice)
            L_is_slice = isinstance(L, slice)
            if g_is_slice and L_is_slice:
                # Then we will have a 2D return type
                # Get start, stop, and size info for our particular data
                g_start, g_stop, g_step = g.indices(self.shape[0])
                g_size = (g_stop - g_start) // g_step

                L_start, L_stop, L_step = L.indices(self.shape[1])
                L_size = (L_stop - L_start) // L_step

                # Allocate storage
                result = np.empty((g_size, L_size))

                i = 0
                for g_idx in range(g_start, g_stop, g_step):
                    if g_idx < self.gout_min or g_idx > self.gout_max:
                        # Then we are in the region that has been compressed
                        # and can confidently return 0s
                        result[i, :] = 0.
                    else:
                        # We have to return the data for the group in question
                        # since there is no funny business with the 2nd-dim
                        # slice, just use it here w/ Numpy.
                        result[i, :] = self.data[g_idx, L]

                    # Increment the location in result
                    i += 1
                return result
            elif g_is_slice:
                # Then we will have a 1D return type
                # Get start, stop, and size info for our particular data
                g_start, g_stop, g_step = g.indices(self.shape[0])
                g_size = (g_stop - g_start) // g_step

                # Allocate storage
                result = np.empty((g_size))

                i = 0
                for g_idx in range(g_start, g_stop, g_step):
                    if g_idx < self.gout_min or g_idx > self.gout_max:
                        # Then we are in the region that has been compressed
                        # and can set to 0
                        result[i] = 0.
                    else:
                        # We have to return the data for the group in question
                        # since there is no funny business with the 2nd-dim
                        # slice, just use it here w/ Numpy.
                        result[i] = self.data[g_idx - self.gout_min, L]

                    # Increment the location in result
                    i += 1
                return result
            elif L_is_slice:
                if g < self.gout_min or g > self.gout_max:
                    # It will be all zeros, but lets figure out how many
                    L_start, L_stop, L_step = L.indices(self.shape[1])
                    L_size = (L_stop - L_start) // L_step
                    result = np.zeros(L_size)
                else:
                    # Just rely on numpy's __getitem__
                    result = self.data[g, L]

                return result
            else:
                # Then we are returning a scalar
                # Check if the requested data is in the range
                if g < self.gout_min or g > self.gout_max:
                    return 0.
                else:
                    # Adjust the group-wise indices for gout_min
                    return self.data[g - self.gout_min, L]

        else:
            raise ValueError("Invalid Indices")

    def __abs__(self):
        result = deepcopy(self)
        result.data = np.abs(result.data)
        return result

    def __lt__(self, other):
        return _generic_compare(self, other, operator.lt)

    def __gt__(self, other):
        return _generic_compare(self, other, operator.gt)

    def __add__(self, other):
        return _generic_addsub(self, other, operator.add)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return _generic_addsub(self, other, operator.sub)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        err = np.seterr(divide='ignore', invalid='ignore')
        result = _generic_muldiv(self, other, operator.mul)
        np.seterr(**err)
        return result

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        err = np.seterr(divide='ignore', invalid='ignore')
        result = _generic_muldiv(self, other, operator.div)
        np.seterr(**err)
        return result

    def __rdiv__(self, other):
        return self.__div__(other)

    def __truediv__(self, other):
        err = np.seterr(divide='ignore', invalid='ignore')
        result = _generic_muldiv(self, other, operator.truediv)
        np.seterr(**err)
        return result

    def __rtruediv__(self, other):
        return self.__div__(other)

    def toarray(self):
        result = np.empty(self.shape)
        result[:self.gout_min, :] = 0.
        result[self.gout_min: self.gout_max + 1, :] = self.data[:, :]
        result[self.gout_max + 1:, :] = 0.
        return result
