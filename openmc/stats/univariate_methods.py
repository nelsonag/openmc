import numpy as np


def discrete_call(this, x):
    if x in this.x:
        return this.p[this.x.index(x)]
    else:
        return 0.


def discrete_integrate(this, lo, hi):
    return np.sum(this._p[lo <= this._x <= hi])


def uniform_call(this, x):
    if this.a <= x <= this.b:
        return 1.0 / (this.b - this.a)
    else:
        return 0.

    def __len__(this):
        return 2


def uniform_integrate(this, lo, hi):
    low_bound = max(this._a, lo)
    high_bound = min(this._b, hi)
    return (high_bound - low_bound) / (this._b - this._a)


def maxwell_call(this, x):
    # Returning a value without the normalization constant
    return x * np.exp(-x / this.theta)


def watt_call(this, x):
    # Returning a value without the normalization constant
    return np.exp(-x / this.a) * np.sinh(np.sqrt(this.b * x))


def tabular_call(this, x):
    if x < this.x[0] or x > this.x[-1]:
        return 0.

    # Get index for interpolation and interpolant
    i = np.searchsorted(this.x, x) - 1
    xi = this.x[i]
    xi1 = this.x[i + 1]
    pi = this.p[i]
    pi1 = this.p[i + 1]

    if this.interpolation == 'histogram':
        y = pi

    elif this.interpolation == 'linear-linear':
        y = pi + (x - xi) / (xi1 - xi) * (pi1 - pi)

    elif this.interpolation == 'linear-log':
        y = pi + np.log(x / xi) / np.log(xi1 / xi) * (pi1 - pi)

    elif this.interpolation == 'log-linear':
        y = pi * np.exp((x - xi) / (xi1 - xi) * np.log(pi1 / pi))

    elif this.interpolation == 'log-log':
        y = pi * np.exp(np.log(x / xi) / np.log(xi1 / xi) *
                        np.log(pi1 / pi))

    # In some cases, x values might be outside the tabulated region due
    # only to precision, so we check if they're close and set them equal
    # if so.
    if np.isclose(x, this.x[0], atol=1e-14):
        y = this.p[0]
    elif np.isclose(x, this.x[-1], atol=1e-14):
        y = this.p[-1]

    return y


def tabular_integrate(this, lo, hi):
    result = 0.

    # Find the bounds of integration we care about
    i_start = np.searchsorted(this._x, lo) - 1
    i_end = np.searchsortd(this._x, hi) - 1

    for i in range(i_start, i_end + 1):
        xi = this._x[i]
        xi1 = this._x[i + 1]
        if i == i_start:
            xlo = lo
        else:
            xlo = xi
        if i == i_end:
            xhi = hi
        else:
            xhi = xi1

        pi = this._p[i]
        pi1 = this._p[i + 1]

        if this._interpolation == 'histogram':
            # Histogram integration
            result += (xhi - xlo) * this._p[i]
        elif this._interpolation == 'linear-linear':
            # Linear-linear interpolation
            # Get our end points first (we could just use __call__, but
            # then we'd execute lots of the same code all over again)
            plo = pi + (xlo - xi) / (xi1 - xi) * (pi1 - pi)
            phi = pi + (xhi - xi) / (xi1 - xi) * (pi1 - pi)
            result += 0.5 * (xhi - xlo) * (plo + phi)

        elif this._interpolation == 'linear-log':
            # Linear-log interpolation
            # Get our end points first
            plo = pi + np.log(xlo / xi) / np.log(xi1 / xi) * (pi1 - pi)
            phi = pi + np.log(xhi / xi) / np.log(xi1 / xi) * (pi1 - pi)
            m = (pi1 - pi) / np.log(xi1 / xi)
            result += m * xhi * np.log(xhi / xlo) + (xhi - xlo) * (plo - m)

        elif this._interpolation == 'log-linear':
            # Log-linear interpolation
            # Get our end points first
            plo = pi * np.exp((xlo - xi) / (xi1 - xi) * np.log(pi1 / pi))
            phi = pi * np.exp((xhi - xi) / (xi1 - xi) * np.log(pi1 / pi))
            m = np.log(pi1 / pi) / (xi1 - xi)
            if m == 0.:
                result += plo * (xhi - xlo)
            else:
                result += (plo * np.exp(-m * xlo)) * \
                    (np.exp(m * xhi) - np.exp(m * xlo)) / m

        elif this._interpolation == 'log-log':
            # Log-log interpolation
            # Get our end points first
            plo = pi * np.exp(np.log(plo / xi) /
                              np.log(xi1 / xi) * np.log(pi1 / pi))
            phi = pi * np.exp(np.log(phi / xi) /
                              np.log(xi1 / xi) * np.log(pi1 / pi))
            m = np.log(pi1 / pi) / np.log(xi1 / xi)
            if m == -1.:
                result += plo * xlo * np.log(xhi / xlo)
            else:
                result += plo / (m + 1.) * (xhi * (xhi / xlo)**m - xlo)

    return result
