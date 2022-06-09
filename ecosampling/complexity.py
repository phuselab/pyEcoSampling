"""Computes spatial configuration complexity.

Authors:
    - Giuseppe Boccignone <giuseppe.boccignone@unimi.it>
    - Renato Nobre <renato.avellarnobre@studenti.unimi.it>

Changes:
    - 12/12/2012  First Edition Matlab
    - 31/05/2022  Python Edition
"""

import numpy as np

from config import ComplexityConfig
from utils.logger import Logger

logger = Logger(__name__)

class Complexity:
    """Handle computation of spatial configuration complexity.

    Note:
        Other complexity algorithm functions can be added here.

    Attributes:
        order (:obj:`list` of :obj:`float`): List of order values
        disorder (:obj:`list` of :obj:`float`): List of disorder values
        complexity(:obj:`list` of :obj:`float`): List of complexity values
        c_type: Complexity algotithm type to execute defined on config.py

    """

    def __init__(self):
        self.order = []
        self.disorder = []
        self.complexity = []
        self.c_type = ComplexityConfig.TYPE

    def compute_complexity(self, histmat, N, n_bins):
        """Computes spatial configuration complexity :math:`C(t)` of Interest points.

        The function is a simple wrapper for complexity computation.
        Executes some kind of complexity algorithm which is defined from the
        class parameter ``self.c_type`` by calling the appropriate function.

        Args:
            histmat (np.ndarray): 2D Spatial histogram of IPs.
            N (float): number of points.
            n_bins (int): number of bins.

        Returns:
            disorder (float): Disorder value.
            order (float): Order value.
            complexity (float): Space complexity value.

        Raises:
            NotImplementedError: If desired complexity type was not implemented.

        Examples:
            >>> disorder, order, compl = esComputeComplexity('SDL', histmat, N, n_bins)
        """
        logger.verbose('Evaluate complexity')
        phistmat = (histmat / N) + np.finfo(float).eps
        H = -np.sum(np.sum(np.multiply(phistmat, np.log(phistmat))))

        if self.c_type == 'SDL':
            order, disorder = self._shiner_davison_landsberg(H, n_bins)
        elif self.c_type == 'LMC':
            order, disorder = self._lopez_ruiz_mancini(H, phistmat, n_bins)
        elif self.c_type == 'FC':
            order, disorder = self._feldman_crutchfield(H, phistmat, n_bins)
        else:
            # Not implemented
            raise NotImplementedError("Unknown complexity type")

        complexity = disorder * order
        self.order.append(order)
        self.disorder.append(disorder)
        self.complexity.append(complexity)

        return order, disorder, complexity

    def _shiner_davison_landsberg(self, H, n_bins):
        """Shiner-Davison-Landsberg (SDL) complexity.

        For more information, see the reference below [1]_.

        Args:
            H (float): Shannon Entropy (Boltzman-Gibbs entropy).
            n_bins (int): number of bins.

        Returns:
            order (float): Order value.
            disorder (float): Disorder value.

        References
        ----------
        .. [1] `Shiner, J. S., Davison, M., & Landsberg, P. T. (1999). Simple measure for complexity.
           Physical review E, 59(2), 1459.
           <https://journals.aps.org/pre/abstract/10.1103/PhysRevE.59.1459>`_
        """
        h_sup = np.log(n_bins)
        disorder = H / h_sup
        order = 1 - disorder
        return order, disorder

    def _lopez_ruiz_mancini(self, H, phistmat, n_bins):
        """Lòpez-Ruiz, Mancini, and Calbet complexity.

        D is called Disequilibrium. This quantity is a measure of the
        divergence of the given probability distribution from the uniform one.
        For more information, see the reference below [2]_.

        Args:
            H (float): Shannon Entropy (Boltzman-Gibbs entropy)
            phistmat (np.ndarray): 2D Spatial histogram of IPs devided by number of points.
            n_bins (int): number of bins.

        Returns:
            order (float): Order value.
            disorder (float): Disorder value.

        References
        ----------
        .. [2] `Lopez-Ruiz, R., Mancini, H. L., & Calbet, X. (1995). A statistical measure of complexity.
           Physics letters A, 209(5-6), 321-326.
           <https://www.sciencedirect.com/science/article/abs/pii/0375960195008675>`_
        """
        D = np.square((phistmat - (1 / n_bins)))
        disorder = H
        order = np.sum(np.sum(D))
        return order, disorder

    def _feldman_crutchfield(self, H, phistmat, n_bins):
        """Feldman and Crutchfield's amendment replaces Order with the Kullback-Leibler divergence.

        For the purpose of serving as a component of complexity, one of the
        compared distributions is taken to be uniform.
        For more information, see the reference below [3]_.

        Args:
            H (float): Shannon Entropy (Boltzman-Gibbs entropy)
            phistmat (np.ndarray): 2D Spatial histogram of IPs devided by number of points.
            n_bins (int): number of bins.

        Returns:
            order (float): Order value.
            disorder (float): Disorder value.

        References
        ----------
        .. [3] `Feldman, D. P., & Crutchfield, J. P. (1998). Measures of statistical complexity: Why?.
           Physics Letters A, 238(4-5), 244-252.
           <https://www.sciencedirect.com/science/article/abs/pii/S0375960197008554>`_
        """
        disorder = H
        order = np.sum(np.sum(phistmat * np.log(n_bins*phistmat)))
        return order, disorder
