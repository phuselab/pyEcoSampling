"""Class file for the Action Selector.

Authors:
    - Giuseppe Boccignone <giuseppe.boccignone@unimi.it>
    - Renato Nobre <renato.avellarnobre@studenti.unimi.it>

Changes:
    - 12/12/2012  First Edition Matlab
    - 31/05/2022  Python Edition
"""

from config import ComplexityConfig
from utils.logger import Logger
from utils.statistics import sample_discrete, sample_dirchlet

logger = Logger(__name__)


class ActionSelector:

    def __init__(self, disorder, order, complexity):
        """Select the new gaze action index based on the complexity.

        Args:
            disorder (float):  disorder parameter
            order (float): order parameter
            complexity (float): complexity parameter $\mathcal{C}(t)$

        Attributes:
            disorder (float): disorder parameter
            order (float): order parameter
            complexity (float): complexity parameter $\mathcal{C}(t)$
            c_eps (float): the chaos edge
        """
        self.c_eps = ComplexityConfig.EPS
        self.complexity = complexity
        self.order = order
        self.disorder = disorder


    def select_action(self, nu):
        """Dirichlet hyper-parameter update.

        Args:
            nu (vector): previous Dirichlet Hyperparameters

        Returns:
            nu (vector): new Dirichlet Hyperparameters
            z (int): selected action index
        """
        nu = self._dirichlet_hyper_param_update(nu)
        logger.verbose(f"Complexity  {self.complexity} // Order {self.order} // Disorder {self.disorder}")
        logger.verbose(f"Parameter nu1 {nu[0]}")
        logger.verbose(f"Parameter nu2 {nu[1]}")
        logger.verbose(f"Parameter nu3 {nu[2]}")

        # Sampling the \pi parameter that is the probability of an order event
        # $$\pi ~ %Dir(\pi | \nu)$$
        pi_prob = sample_dirchlet(nu, 1)

        # Sampling the kind of gaze-shift regime:
        # $$ z ~ Mult(z | \pi) $$
        z = sample_discrete(pi_prob, 1, 1)
        logger.verbose(f"Action sampled: z = {z}")

        return nu, z

    def _dirichlet_hyper_param_update(self, nu):
        """Dirichlet hyper-parameter update.

        Computes the new Dirichlet hyper-parameter nu. Given the complexity Ct,
        we partition the complexity range in order to define K possible
        complexity events. This way the hyper-parameter update can be rewritten
        as the recursion. For further information, see [1]_ and [2]_.

        Args:
            nu (vector): old Dirichlet Hyperparameters

        Returns:
            nu (vector): new Dirichlet Hyperparameters

        Examples:
            >>> nu = _dirichlet_hyper_param_update(nu_old, disorder, order, compl, COMPL_EDGE)

        References
        ----------
        .. [1] `Boccignone, G., & Ferraro, M. (2013). Ecological sampling of gaze shifts.
           IEEE transactions on cybernetics, 44(2), 266-279.
           <https://ieeexplore.ieee.org/abstract/document/6502674>`_
        .. [2] `G. Boccignone and M. Ferraro, The active sampling of gaze-shifts,
           in Image Analysis and Processing ICIAP 2011, ser. Lecture Notes in Computer Science,
           G. Maino and G. Foresti, Eds. Springer Berlin / Heidelberg, 2011, vol. 6978, pp. 187?196.
           <https://ieeexplore.ieee.org/abstract/document/6502674>`_
        """

        thresh = (0.25 - self.c_eps) # If beyond threshold we are in the complex domain
        reset_step = 25 # Parameter to control hysteresis

        for k in range(len(nu)):
            nu[k] = nu[k] % reset_step
            if nu[k] == 0:
                nu[k] = 1

        if self.complexity <= thresh:
            if (self.disorder < self.order):
                # Order Event
                nu[0] = nu[0] + 1
            else:
                # Disorder Event
                nu[2] = nu[2] + 1
        else:
            # At the edge of chaos
            nu[1] = nu[1] + 1
            logger.verbose('MAX COMPLEXITY!!!!!!')

        return nu
