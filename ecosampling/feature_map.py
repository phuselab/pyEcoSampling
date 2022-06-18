"""Feature Map class file.

Authors:
    - Giuseppe Boccignone <giuseppe.boccignone@unimi.it>
    - Renato Nobre <renato.avellarnobre@studenti.unimi.it>

Changes:
    - 12/12/2012  First Edition Matlab
    - 31/05/2022  Python Edition
"""


import numpy as np

from config import GeneralConfig, SaliencyConfig
from utils.logger import Logger
from backends.self_resemblance import SelfRessemblance

logger = Logger(__name__)

class FeatureMap:
    """Compute features and create feature map.

    The compute features is a simple wrapper for feature computation.
    Executes the feature extraction algorithm which is defined.

    Note:
        We implemented as a backend the Static and space-time visual
        saliency detection by self-resemblance method. Additional methods
        require to be implemented.

    Attributes:
        feature_type (str): Experiment type for saliency computation.
        wsize (int): LARK spatial window size.
        wsize_t (int): LARK temporal window size.
        alpha (float): LARK sensitivity parameter.
        sigma (float): Fall-off parameter for self-resemblamnce.
        h (float): Smoothing parameter for LARK.
        show (np.ndarray): Version of the feature map to visualization.
    """


    def __init__(self):
        self.feature_type = GeneralConfig.EXPERIMENT_TYPE
        self.wsize = SaliencyConfig.WSIZE
        self.wsize_t = SaliencyConfig.WSIZE_T
        self.alpha = SaliencyConfig.LARK_ALPHA
        self.sigma = SaliencyConfig.LARK_SIGMA
        self.h = SaliencyConfig.LARK_H
        self.show = None

    def compute_features(self, fov_seq, frame_sampling):
        """Computes features using a foveated sequence of frames.

        The function is a simple wrapper for feature computation. Executes some kind
        of feature extraction algorithm which is defined from the
        ``feature_type`` by calling the appropriate function.

        Note:
            Here for simplicity only the Self Resemblance method has been considered.
            If other methods need to be experimented, then you should extend the
            if...elif... control structure. For further information, see also [1]_.

        Args:
            fov_seq (matrix): the foveated sequence of frames.
            feature_type (string): the chosen method.
            feature_params (dict): the parameters for the chosen feature.

        Returns:
            fmap (matrix): the feature map.

        Examples:
            >>> fMap = esComputeFeatures(fov_seq, '3DLARK_SELFRESEMBLANCE', feature_params)

        References
        ----------
        .. [1] `Seo, H. J., & Milanfar, P. (2009). Static and space-time visual saliency detection
           by self-resemblance. Journal of vision, 9(12), 15-15.
           <https://jov.arvojournals.org/article.aspx?articleid=2122209>`_
        """
        logger.verbose("Get features")
        feature_map = None
        if self.feature_type == '3DLARK_SELFRESEMBLANCE':
            feature_method = SelfRessemblance(self.wsize, self.wsize_t, self.alpha, self.sigma, self.h)
            feature_map = feature_method.three_D_LARK(fov_seq)
        else:
            raise NotImplementedError("UNKNOWN TYPE OF EXPERIMENT")

        self.show = frame_sampling.frame_resize_orginal(feature_map[:,:,1,1].astype('double'))

        return feature_map
