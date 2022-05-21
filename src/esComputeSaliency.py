# function sMap = esComputeSalience(fMap, Seq, salType, sParam)
# %esComputeSalience - a wrapper for salience computation
# %
# % Synopsis
# %   sMap = esComputeSalience(fMap, Seq, salType, sParam)
# %
# % Description
# %   The function is a simple wrapper for salience computation. Executes some kind
# %   of salience computation algorithm which is defined from the parameter
# %   salType by calling the appropriate function. Here for simplicity only
# %   the 3-D SELF RESEMBLANCE SPATIO TEMPORAL SALIENCY method has been considered.
# %
# %   If other methods need to be experimented, then you should extend the if...elseif...end
# %   control structure
# %   NOTE: any kind of salience or oriority map computation will do:
# %   bottom-up, top-down etc.
# %
# % Inputs ([]s are optional)
# %   (matrix) fMap         the foveated feature map
# %   (matrix) Seq          the foveated sequence of frames
# %   (string) salType      salience computation method
# %   (struct) sParam       salience computation parameters
# %
# % Outputs ([]s are optional)
# %   (matrix) Sal          the salience map
# %   ....
# %
# % Examples
# %   sMap  = esComputeSalience(foveated_fMap, Seq, '3DLARK_SELFRESEMBLANCE', sParam);
# %
# % See also
# %   SpaceTimeSaliencyMap
# %
# % Requirements
# %   SpaceTimeSaliencyMap (./saltools/SelfResemblance2/)

# % References
# %   H. Seo and P. Milanfar, Static and space-time visual saliency detection by self-resemblance,
# %                           Journal of Vision, vol. 9, no. 12, pp. 1?27, 2009
# %
# %
# % Authors
# %   Giuseppe Boccignone <Giuseppe.Boccignone(at)unimi.it>
# %
# % License
# %   The program is free for non-commercial academic use. Please
# %   contact the authors if you are interested in using the software
# %   for commercial purposes. The software must not modified or
# %   re-distributed without prior permission of the authors.
# %
# % Changes
# %   12/12/2012  First Edition
# %

# if strcmp(salType,'3DLARK_SELFRESEMBLANCE')
#     % Compute 3-D SELF RESEMBLANCE SPATIO TEMPORAL SALIENCY
#     SM   = SpaceTimeSaliencyMap(Seq,fMap,sParam.wsize,sParam.wsize_t,sParam.sigma);
#     % Salience on the current frame
#     sMap = SM(:,:,2);
# else
#     fprintf('\n UNKNOWN SALIENCE COMPUTATION TYPE.....');
# end

# end
