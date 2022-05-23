

# % See also
# %   SpaceTimeSaliencyMap
# %
# % Requirements
# %   SpaceTimeSaliencyMap (./saltools/SelfResemblance2/)
# %
# % References
# %   H. Seo and P. Milanfar, Static and space-time visual saliency detection by self-resemblance,
# %                           Journal of Vision, vol. 9, no. 12, pp. 1?27, 2009
# %
# %
# % Authors
# %   Giuseppe Boccignone <Giuseppe.Boccignone(at)unimi.it>
# %
# %
# % Changes
# %   12/12/2012  First Edition
# %

def esComputeSalience(f_map, seq, sal_type, s_param):
    """A wrapper for salience computation.

    The function is a simple wrapper for salience computation. Executes some kind
    of salience computation algorithm which is defined from the parameter
    sal_type by calling the appropriate function. Here for simplicity only
    the 3-D SELF RESEMBLANCE SPATIO TEMPORAL SALIENCY method has been considered.

    Args:
        f_map (matrix): the foveated feature map
        seq (matrix): the foveated sequence of frames
        sal_type (string): the salience computation method
        s_param (struct): the salience computation parameters

    Outputs:
        s_map (matrix): the salience map on the current frame

    Note:
        Any kind of salience or oriority map computation will do:
         bottom-up, top-down, etc.If other methods need to be experimented,
        then you should extend the control structure.

    Examples:
        >>> sMap  = esComputeSalience(foveated_fMap, seq, '3DLARK_SELFRESEMBLANCE', s_param);

    """
    if sal_type == '3DLARK_SELFRESEMBLANCE':
        # Compute 3-D SELF RESEMBLANCE SPATIO TEMPORAL SALIENCY
        sm = SpaceTimeSaliencyMap(seq, f_map, s_param["wsize"], s_param["wsize_t"], s_param["sigma"])
        # Salience on the current frame
        s_map = sm[:,:,2]
        return s_map
    else:
        print('\n UNKNOWN SALIENCE COMPUTATION TYPE.....')
