import numpy as np
from config import ComplexityConfig
from utils.logger import Logger

logger = Logger(__name__)


class Complexity:

    def __init__(self):
        self.order = []
        self.disorder = []
        self.complexity = []
        self.c_type = ComplexityConfig.TYPE

    # % References
    # %  J. Shiner, M. Davison, and P. Landsberg, ?Simple measure for complexity,?
    # %  Physical review E, vol. 59, no. 2, pp. 1459?1464, 1999.
    # %
    # %  R. Lopez-Ruiz, H. Mancini, and X. Calbet, ?A statistical measure of complexity* 1,?
    # %  Physics Letters A, vol. 209, no. 5-6, pp. 321?326, 1995.
    # %
    # %  D. Feldman and J. Crutchfield, ?Measures of statistical complexity: Why??
    # %  Physics Letters A, vol. 238, no. 4-5, pp.244?252, 1998
    # %
    # % Authors
    # %   Giuseppe Boccignone <Giuseppe.Boccignone(at)unimi.it>
    # %
    # % Changes
    # %   12/12/2012  First Edition



    def compute_complexity(self, histmat, N, n_bins):
        """Computes spatial configuration complexity $$ C(t)$$ of Interest points.

        The function is a simple wrapper for complexity computation.
        Executes some kind of complexity algorithm which is defined from the
        parameter c_type by calling the appropriate function.

        Args:
            c_type (string): Chosen complexity method.
            histmat (matrix): 2D Spatial histogram of IPs.
            N (float): number of points.
            n_bins (float): number of bins.

        Returns:
            disorder (float): Disorder value.
            order (float): Order value.
            complexity (float): Space complexity value.

        Examples:
            >>> disorder, order, compl = esComputeComplexity('SDL', histmat, N, n_bins)
        """
        logger.verbose('Evaluate complexity')
        # H is Shannon entropy (which is an equivalent of Boltzman-Gibbs's entropy)
        phistmat = (histmat / N) + np.finfo(float).eps
        H = -np.sum(np.sum(np.multiply(phistmat, np.log(phistmat))))

        if self.c_type == 'SDL':
            # Shiner-Davison-Landsberg (SDL) complexity
            order, disorder = self._shiner_davison_landsberg(H, n_bins)
        elif self.c_type == 'LMC':
            # Lòpez-Ruiz, Mancini, and Calbet complexity
            order, disorder = self._lopez_ruiz_mancini(H, phistmat, n_bins)
        elif self.c_type == 'FC':
            # Feldman and Crutchfield's amendment replaces Order with the Kullback-Leibler divergence.
            order, disorder = self._feldman_crutchfield(H, phistmat, n_bins)
        else:
            # Not implemented
            print("UNKNOWN TYPE OF COMPLEXITY")

        complexity = disorder * order
        self.order.append(order)
        self.disorder.append(disorder)
        self.complexity.append(complexity)


    def _shiner_davison_landsberg(self, H, n_bins):
        h_sup = np.log(n_bins)
        disorder = H / h_sup
        order = 1 - disorder
        return order, disorder

    def _lopez_ruiz_mancini(self, H, phistmat, n_bins):
        # D is called Disequilibrium. This quantity is a measure of the
        # divergence of the given probability distribution from the uniform one.
        D = np.square((phistmat - (1 / n_bins)))
        disorder = H
        order = np.sum(np.sum(D))
        return order, disorder

    def _feldman_crutchfield(self, H, phistmat, n_bins):
        # For the purpose of serving as a component of complexity, one of the
        # compared distributions is taken to be uniform
        disorder = H
        order = np.sum(np.sum(phistmat * np.log(n_bins*phistmat)))
        return order, disorder
