import numpy as np
import scipy.special as ss

from .integrators import integrate


def linearizer_wrapper(linearizer_args):
    # multiprocessing.Pool requires a single argument to be passed for python2
    # compatibility; so this method acts as the interface between
    # multiprocessing.Pool and _linearizer
    return _linearizer(*linearizer_args)


def _linearizer(Ein_grid, func, args, tolerance):
    # This method constructs the Ein and results grid with the incoming grid
    # set so that the tolerance is achieved when interpolating between values
    # This method is based on the openmc.data.linearize method, but modified to
    # accept arguments to the function and to also work on a numpy.ndarray
    # result from the function.

    # Initialize output arrays
    Ein_output = []
    results_out = []

    # Initialize stack
    Ein_stack = [Ein_grid[0]]
    results_stack = [func(Ein_grid[0], *args)]
    error = np.empty(results_stack[0].shape)

    for i in range(Ein_grid.shape[0] - 1):
        Ein_stack.insert(0, Ein_grid[i + 1])
        results_stack.insert(0, func(Ein_grid[i + 1], *args))

        while True:
            Ein_high, Ein_low = Ein_stack[-2:]
            results_high, results_low = results_stack[-2:]
            Ein_mid = 0.5 * (Ein_low + Ein_high)
            results_mid = func(Ein_mid, *args)

            results_interp = results_low + (results_high - results_low) / \
                (Ein_high - Ein_low) * (Ein_mid - Ein_low)

            error = np.subtract(results_interp, results_mid)
            # Avoid division by 0 errors since they are fully expected
            # with our sparse results
            with np.errstate(divide='ignore', invalid='ignore'):
                error = np.abs(np.nan_to_num(np.divide(error, results_mid)))

            if np.any(error > tolerance):
                Ein_stack.insert(-1, Ein_mid)
                results_stack.insert(-1, results_mid)
            else:
                Ein_output.append(Ein_stack.pop())
                results_out.append(results_stack.pop())
                if len(Ein_stack) == 1:
                    break

    Ein_output.append(Ein_stack.pop())
    results_out.append(results_stack.pop())

    return np.array(Ein_output), np.array(results_out)


def do_sab(Ein, elastic_args, inelastic_args, inelastic_distribution):
    if inelastic_distribution:
        results = integrate(Ein, *inelastic_args)
    else:
        results = _do_sab_inelastic(Ein, *inelastic_args)
    if elastic_args:
        elastic = _do_sab_elastic(Ein, *elastic_args)
        results = np.add(results, elastic)

    return results


def _do_sab_elastic(Ein, num_groups, num_angle, scatter_format, group_edges,
                    coherent, mu_out, xs_Ein, xs, mu_bins):
    result = np.zeros((num_groups, num_angle))

    if Ein < xs_Ein[0] or Ein > xs_Ein[-1]:
        return result

    # For elastic scattering the incoming energy is the same as the outgoing
    # energy, so use the incoming energy to find the outgoing energy group
    g = np.searchsorted(group_edges, Ein) - 1

    # Now compute the integral
    if coherent:
        # Find relevant values of mu and the structure factor
        # (i.e., the ones where mu_i > -1)
        mu_is = []
        b_i_over_E = []
        for i in range(len(xs.bragg_edges)):
            mu_i = 1. - 2. * xs.bragg_edges[i] / Ein
            if mu_i >= -1.:
                mu_is.append(mu_i)
                b_i_over_E.append(xs.factors[i] / Ein)

        if scatter_format == 'legendre':
            for l in range(num_angle):
                result[g, l] = np.sum(np.multiply(ss.eval_legendre(l, mu_is),
                                                  b_i_over_E))
        else:
            for u in range(len(mu_bins[:-1])):
                for mu_i in mu_is:
                    if mu_bins[u] <= mu_i < mu_bins[u + 1]:
                        result[g, u] += b_i_over_E[u]
    else:
        data = mu_out
        j = np.searchsorted(xs_Ein, Ein) - 1
        f = (Ein - xs_Ein[j]) / (xs_Ein[j + 1] - xs_Ein[j])
        wgt = 1.0 / float(data.shape[1])
        if scatter_format == 'legendre':
            for imu in range(data.shape[1]):
                mu = (1. - f) * data[j, imu] + f * data[j + 1, imu]
                for l in range(num_angle):
                    result[g, l] += wgt * ss.eval_legendre(l, mu)
        else:
            for imu in range(data.shape[1]):
                mu = (1. - f) * data[j, imu] + f * data[j + 1, imu]
                k = np.digitize(mu, mu_bins) - 1
                result[g, k] += wgt
        result[g, :] *= xs(Ein)

    return result


def _do_sab_inelastic(Ein, num_groups, num_angle, scatter_format, group_edges,
                      Eout_data, mu_data, mu_bins, wgt, xs):
    result = np.zeros((num_groups, num_angle))

    if Ein < xs._x[0] or Ein > xs._x[-1]:
        return result

    j = np.searchsorted(xs._x, Ein) - 1
    f = (Ein - xs._x[j]) / (xs._x[j + 1] - xs._x[j])

    # Get the cross section value
    xs_val = xs(Ein)

    # Loop through each equally likely Eout, find its group
    # and sum legendre moments of discrete mu to it.
    # Eout and mu need to be interpolated to
    for eo in range(Eout_data.shape[1]):
        # Interpolate to our Eout
        Eout = (1. - f) * Eout_data[j, eo] + \
            f * Eout_data[j + 1, eo]
        if Eout < group_edges[0] or Eout >= group_edges[-1]:
            return result

        g = np.digitize(Eout, group_edges) - 1

        # Find interpolated mu values
        mus = np.add(np.multiply((1. - f), mu_data[j, eo, :]),
                     np.multiply(f, mu_data[j + 1, eo, :]))

        wgt_xs = wgt[eo] * xs_val

        # Find the bins in our histogram of the mus, if necessary
        if scatter_format != 'legendre':
            ks = np.digitize(mus, mu_bins) - 1
            for u in range(mu_data.shape[2]):
                result[g, ks[u]] += wgt_xs
        else:
            for u in range(mu_data.shape[2]):
                for l in range(result.shape[-1]):
                    result[g, l] += wgt_xs * ss.eval_legendre(l, mus[u])

    return result


def do_neutron_scatter(Ein, awr, rxns, products, this, kT, mu_bins, xs_func,
                       method, mus_grid=None, wgts=None):
    # Initialize the storage
    results = np.zeros((this.num_groups, this.num_angle))

    # Get results for each reaction
    for r, rxn in enumerate(rxns):
        xs = xs_func[r]
        product = products[r]

        # Dont waste time if not above the threshold energy
        if Ein <= xs._x[0]:
            continue

        yield_xs = product.yield_(Ein) * xs(Ein)
        # If the yield is 0, also dont waste time
        if yield_xs <= 0.:
            continue

        rxn_results = np.zeros((this.num_groups, this.num_angle))
        for d, distrib in enumerate(product.distribution):
            if len(product.applicability) > 1:
                applicability = product.applicability[d](Ein)
            else:
                applicability = 1.

            unnorm_result = integrate(Ein, distrib, this.group_edges,
                                      this.scatter_format, rxn.center_of_mass,
                                      awr, this.freegas_cutoff * kT, kT,
                                      rxn.q_value, mu_bins, this.order, xs,
                                      method, mus_grid, wgts)

            # Now normalize and multiply by the yield, xs, and applicability
            if this.scatter_format == 'legendre':
                norm_factor = np.sum(unnorm_result[:, 0])
            else:
                norm_factor = np.sum(unnorm_result)
            if norm_factor > 0.:
                rxn_results[:, :] += \
                    applicability * unnorm_result / norm_factor

        results += yield_xs * rxn_results

    return results


def do_chi(Ein, awr, rxns, this, kT, xs_func, n_delayed):
    # Initialize the storage; the first dimension is for prompt/delayed
    # and the second is for total
    results = np.zeros((2 + n_delayed, this.num_groups))

    # Get results for each reaction
    delay_index = 0
    for r, rxn in enumerate(rxns):
        if Ein < xs_func[r]._x[0]:
            continue

        xs = xs_func[r](Ein)

        for product in rxn.products:
            if product.particle != 'neutron':
                continue
            if product.emission_mode == 'prompt':
                index = 1
            else:
                index = 2 + delay_index
                delay_index += 1

            yield_xs = product.yield_(Ein) * xs
            # If the yield is 0, also dont waste time
            if yield_xs <= 0.:
                continue

            rxn_results = np.zeros(this.num_groups)

            for d, distrib in enumerate(product.distribution):
                if len(product.applicability) > 1:
                    applicability = product.applicability[d](Ein)
                else:
                    applicability = 1.

                unnorm_result = \
                    integrate(Ein, distrib, this.group_edges, 'histogram',
                              rxn.center_of_mass, awr, this.freegas_cutoff,
                              kT, rxn.q_value)[:, 0]
                norm_factor = np.sum(unnorm_result)
                if norm_factor > 0.:
                    rxn_results += \
                        applicability * unnorm_result / norm_factor

            results[0, :] += yield_xs * rxn_results

            results[index, :] += rxn_results
    return results
