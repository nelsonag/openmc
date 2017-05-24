from collections import Iterable

import numpy as np


def tabulated1d_call(this, x):
    # Check if input is array or scalar
    if isinstance(x, Iterable):
        iterable = True
        x = np.array(x)
    else:
        iterable = False
        x = np.array([x], dtype=float)

    # Create output array
    y = np.zeros_like(x)

    # Get indices for interpolation
    idx = np.searchsorted(this.x, x, side='right') - 1

    # Loop over interpolation regions
    for k in range(len(this.breakpoints)):
        # Get indices for the begining and ending of this region
        i_begin = this.breakpoints[k - 1] - 1 if k > 0 else 0
        i_end = this.breakpoints[k] - 1

        # Figure out which idx values lie within this region
        contained = (idx >= i_begin) & (idx < i_end)

        xk = x[contained]                 # x values in this region
        xi = this.x[idx[contained]]       # low edge of corresponding bins
        xi1 = this.x[idx[contained] + 1]  # high edge of corresponding bins
        yi = this.y[idx[contained]]
        yi1 = this.y[idx[contained] + 1]

        if this.interpolation[k] == 1:
            # Histogram
            y[contained] = yi

        elif this.interpolation[k] == 2:
            # Linear-linear
            y[contained] = yi + (xk - xi)/(xi1 - xi)*(yi1 - yi)

        elif this.interpolation[k] == 3:
            # Linear-log
            y[contained] = yi + np.log(xk/xi)/np.log(xi1/xi)*(yi1 - yi)

        elif this.interpolation[k] == 4:
            # Log-linear
            y[contained] = yi*np.exp((xk - xi)/(xi1 - xi)*np.log(yi1/yi))

        elif this.interpolation[k] == 5:
            # Log-log
            y[contained] = (yi*np.exp(np.log(xk/xi)/np.log(xi1/xi)
                            *np.log(yi1/yi)))

    # In some cases, x values might be outside the tabulated region due only
    # to precision, so we check if they're close and set them equal if so.
    y[np.isclose(x, this.x[0], atol=1e-14)] = this.y[0]
    y[np.isclose(x, this.x[-1], atol=1e-14)] = this.y[-1]

    return y if iterable else y[0]
