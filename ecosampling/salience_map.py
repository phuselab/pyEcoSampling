"""Salience Map class file.

Authors:
    - Giuseppe Boccignone <giuseppe.boccignone@unimi.it>
    - Renato Nobre <renato.avellarnobre@studenti.unimi.it>

Changes:
    - 12/12/2012  First Edition Matlab
    - 31/05/2022  Python Edition
"""


from config import GeneralConfig, SaliencyConfig
from utils.helper import EdgeMirror3
import numpy as np
from utils.logger import Logger

logger = Logger(__name__)

class SalienceMap:
    """Compute salience map given a feature map.

    The salience map is a simple wrapper for salience computation.
    Executes the salience algorithm which is defined.

    Note:
        We implemented as a backend the Static and space-time visual
        saliency detection by self-resemblance method. Additional methods
        require to be implemented.

    Attributes:
        sal_type (str): Experiment type for saliency computation.
        wsize (int): LARK spatial window size.
        wsize_t (int): LARK temporal window size.
        sigma (float): LARK fall-off parameter.
        show (matrix): Saliency map on the current frame.
    """

    def __init__(self):
        self.sal_type = GeneralConfig.EXPERIMENT_TYPE
        self.wsize = SaliencyConfig.WSIZE
        self.wsize_t = SaliencyConfig.WSIZE_T
        self.sigma = SaliencyConfig.LARK_SIGMA
        self.show = None

    def compute_salience(self, feature_map, frame_sampling):
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

        Returns:
            s_map (matrix): the salience map on the current frame

        Raises:
            NotImplementedError: if the salience computation method is not implemented.

        Note:
            Any kind of salience or oriority map computation will do:
            bottom-up, top-down, etc.If other methods need to be experimented,
            then you should extend the control structure.

        Examples:
            >>> sMap  = esComputeSalience(foveated_fMap, seq, '3DLARK_SELFRESEMBLANCE', s_param);

        """
        logger.verbose('Sample a saliency map')
        s_map = None
        if self.sal_type == '3DLARK_SELFRESEMBLANCE':
            # Compute 3-D SELF RESEMBLANCE SPATIO TEMPORAL SALIENCY
            sm = self._space_time_saliency_map(feature_map)
            # Salience on the current frame
            s_map = sm[:,:,2]
            s_map = frame_sampling.frame_resize_orginal(s_map[:,])
        else:
            raise NotImplementedError('UNKNOWN SALIENCE COMPUTATION TYPE')

        self.show = s_map
        return s_map



    def _space_time_saliency_map(self, lark):
        """Compute Space-time Self-Resemblance.

        Note:
            Adapted from Matlab's version by Hae Jong on Apr 25, 2011 [1]

        Args:
            lark (np.array): 3D LARK descriptors

        Returns:
            Space-time Saliency Map

        References
        ----------
        .. [1] `Seo, H. J., & Milanfar, P. (2009). Static and space-time visual saliency detection
           by self-resemblance. Journal of vision, 9(12), 15-15.
           <https://jov.arvojournals.org/article.aspx?articleid=2122209>`_
        """
        wsize = self.wsize
        size_t = self.wsize_t
        sigma = self.sigma

        win = (wsize-1) // 2
        win_t = (size_t-1) // 2

        width = np.array([win,win,win_t])

        ls0, ls1, ls2, ls3 = lark.shape

        # To avoid edge effect, we use mirror padding.
        lark1 = np.zeros((ls0+2*win, ls1+2*win, ls2+2*win_t, ls3))
        for i in range(0, lark.shape[3]):
            lark1[:,:,:,i] = EdgeMirror3(lark[:,:,:,i], width)

        # Precompute Norm of center matrices and surrounding matrices
        norm_C = np.zeros((ls0, ls1, ls2))
        for i in range(0, ls0):
            for j in range(0, ls1):
                for l in range(0, ls2):
                    norm_C[i,j,l] = np.linalg.norm(np.squeeze(lark[i,j,l,:]))

        norm_S = np.zeros((ls0+2*win, ls1+2*win, ls2+2*win_t))
        norm_S[:,:,:] = EdgeMirror3(norm_C, width)

        new_shape = [ls0*ls1*ls2, ls3]
        center = np.reshape(lark, new_shape)
        new_shape[1] = 1
        norm_C = np.reshape(norm_C, new_shape)

        saliency_map = np.zeros(norm_C.shape)

        for i in range(0, wsize):
            for j in range(0, wsize):
                for l in range(0, size_t):
                    new_shape[1] = ls3
                    # LARK1(i:i+size(LARK,1)-1,j:j+size(LARK,2)-1,l:l+size(LARK,3)-1,:)
                    lark_reshaped = np.reshape(lark1[i:i+ls0,j:j+ls1,l:l+ls2,:], new_shape)
                    temp = np.sum(np.multiply(center, lark_reshaped), axis=1)
                    # Compute inner product between a center and surrounding matrices
                    new_shape[1] = 1
                    # norm_S(i:i+size(LARK,1)-1,j:j+size(LARK,2)-1,l:l+size(LARK,3)-1)
                    norm_s_reshaped = np.reshape(norm_S[i:i+ls0, j:j+ls1, l:l+ls2], new_shape)

                    temp = np.divide(np.expand_dims(temp, axis=1), np.multiply(norm_C, norm_s_reshaped))

                    # compute self-resemblance using matrix cosine similarity
                    saliency_map = np.add(saliency_map, np.exp(np.divide((-1+temp), np.square(sigma))))

        # Final saliency map values
        saliency_map = np.divide(1, saliency_map)

        new_shape = [ls0, ls1, ls2]
        saliency_map = np.reshape(saliency_map, new_shape)
        return saliency_map

