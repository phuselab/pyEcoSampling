
# % See also
# %   ThreeDLARK
# %
# % Requirements
# %   ThreeDLARK (./saltools/SelfResemblance2/)
# %
# % References
# %   H. Seo and P. Milanfar, Static and space-time visual saliency detection by self-resemblance,
# %                           Journal of Vision, vol. 9, no. 12, pp. 1?27, 2009
# %
# % Authors
# %   Giuseppe Boccignone <Giuseppe.Boccignone(at)unimi.it>
# %
# % Changes
# %   12/12/2012  First Edition
# %




def esComputeFeatures(fov_seq, feature_type, feature_params):
    """Computes features using a foveated sequence of frames.

    The function is a simple wrapper for feature computation. Executes some kind
    of feature extraction algorithm which is defined from the parameter
    feature_type by calling the appropriate function.

    Args:
        fov_seq (matrix): the foveated sequence of frames.
        feature_type (string): the chosen method.
        feature_params (dict): the parameters for the chosen feature.

    Returns:
        fmap (matrix): the feature map.

    Examples:
        >>> fMap = esComputeFeatures(fov_seq, '3DLARK_SELFRESEMBLANCE', feature_params)

    Note:
        Here for simplicity only the Self Resemblance method has been considered.
        If other methods need to be experimented, then you should extend the if...elseif...end
        control structure.
    """

    if feature_type == '3DLARK_SELFRESEMBLANCE':
        fmap = ThreeDLARK(fov_seq, feature_params)
        return fmap
    else:
        print("UNKNOWN TYPE OF EXPERIMENT")

