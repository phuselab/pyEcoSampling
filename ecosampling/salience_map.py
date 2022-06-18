"""Salience Map class file.

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
        self.alpha = SaliencyConfig.LARK_ALPHA
        self.sigma = SaliencyConfig.LARK_SIGMA
        self.h = SaliencyConfig.LARK_H
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
            salience_method = SelfRessemblance(self.wsize, self.wsize_t, self.alpha, self.sigma, self.h)
            sm = salience_method.space_time_saliency_map(feature_map)
            # Salience on the current frame
            s_map = sm[:,:,1]
            s_map = frame_sampling.frame_resize_orginal(s_map[:,])
        else:
            raise NotImplementedError('UNKNOWN SALIENCE COMPUTATION TYPE')

        self.show = s_map
        return s_map
