"""Collection of non-specific helper functions.

Authors:
    - Giuseppe Boccignone <giuseppe.boccignone@unimi.it>
    - Renato Nobre <renato.avellarnobre@studenti.unimi.it>

Changes:
    - 12/12/2012  First Edition Matlab
    - 31/05/2022  Python Edition
"""

import numpy as np

def EdgeMirror3(x, width):
    """Pad with mirroring the edges of the image.s

    Pads with the reflection of the vector mirrored
    on the first and last values of the vector along each axis.

    Note:
        Adapted from Matlab's version by Hae Jong on Apr 25, 2011 [1]_.

    Args:
        x (np.ndarray): Image to be reflected.
        width (vector): Width for each axis of the padding.

    Returns:
        z: Mirror padded image.

    References
    ----------
    .. [1] `Seo, H. J., & Milanfar, P. (2009). Static and space-time visual saliency detection
        by self-resemblance. Journal of vision, 9(12), 15-15.
        <https://jov.arvojournals.org/article.aspx?articleid=2122209>`_
    """
    width = width.astype(int)
    end = -1
    y = np.concatenate((x[:, 2:width[1]:-1,:], x, x[: ,end:end-width[1]:-1,:]), axis=1)
    y = np.concatenate((y[2:width[0]:-1, :,:], y, y[end:end-width[0]:-1, :,:]), axis=0)
    z = np.concatenate((y[:, :, 2:width[2]:-1], y, y[:, :, end:end-width[2]:-1]), axis=2)
    return z


import numpy as np

def mk_gaussian(size, cov=None, mean=None, amplitude='norm'):
    """Create a gaussian filter matrix.

    Compute a matrix with dimensions ``size`` (a [Y X] 2-vector, or a
    scalar) containing a Gaussian function, centered at pixel position
    specified by ``mean`` (default = (size+1)/2), with given ``cov`` (can
    be a scalar, 2-vector, or 2x2 matrix.  Default = (min(size)/6)^2),
    and ``amplitude``. ``amplitude='norm'`` (default) will produce a
    probability-normalized function.

    Notes:
        Function adapted from Matlab's Eero Simoncelli, 6/96.

    Args:
        size (np.ndarray): Dimensions of the matrix.
        cov (np.ndarray, optional): Covariance of the matrix. Defaults to None.
        mean (_type_, optional): Mean of the matrix. Defaults to None.
        amplitude (str, optional): Amplitude type. Defaults to 'norm'.

    Returns:
        Matrix with gausian function centered.
    """

    size = size.flatten('F')
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
