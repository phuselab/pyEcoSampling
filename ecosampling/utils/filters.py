# % IM = mkGaussian(SIZE, COVARIANCE, MEAN, AMPLITUDE)
# %
# % Compute a matrix with dimensions SIZE (a [Y X] 2-vector, or a
# % scalar) containing a Gaussian function, centered at pixel position
# % specified by MEAN (default = (size+1)/2), with given COVARIANCE (can
# % be a scalar, 2-vector, or 2x2 matrix.  Default = (min(size)/6)^2),
# % and AMPLITUDE.  AMPLITUDE='norm' (default) will produce a
# % probability-normalized function.  All but the first argument are
# % optional.

# % Eero Simoncelli, 6/96.

import numpy as np

def mk_gaussian(size, cov=None, mean=None, amplitude='norm'):

    size = size[:]
    if len(size) == 1:
        size = np.array([size, size])

    if cov is None:
        cov = np.square(np.min(size)/6)

    if mean is None:
        mean = (size+1)/2

    x_ramp, y_ramp = np.meshgrid(np.arange(1, size[1]+1) - mean[1], np.arange(1, size[0]+1) - mean[0])


    if np.isscalar(cov):  # Scalar
        if amplitude == 'norm':
            amplitude = 1/(2*np.pi*cov[1])
        e = (np.square(x_ramp) + np.square(y_ramp))/(-2 * cov)
    elif cov.ndim == 2: #  2D-Vector
        if amplitude == 'norm':
            amplitude = 1/(2*np.pi*np.sqrt(cov[0]*cov[1]))
        e = np.square(x_ramp)/(-2 * cov[1]) + np.square(y_ramp)/(-2 * cov[0])
    else:
        if amplitude == 'norm':
            amplitude = 1/(2 * np.pi * np.sqrt(np.linalg.det(cov)))
        cov = -np.linalg.inv(cov)/2
        e = cov[1,1]*np.square(x_ramp) + (cov[0,1]+cov[1,0]) * np.multiply(x_ramp, y_ramp) + cov[0,0]*np.square(y_ramp)

    return np.multiply(amplitude, np.exp(e))
