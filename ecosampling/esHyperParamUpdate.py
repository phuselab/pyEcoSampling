# % References
# %   [1] G. Boccignone and M. Ferraro, The active sampling of gaze-shifts, in Image Analysis and Processing ICIAP 2011,
# %                                     ser. Lecture Notes in Computer Science, G. Maino and G. Foresti, Eds.
# %                                     Springer Berlin / Heidelberg, 2011, vol. 6978, pp. 187?196
# %
# %   [2] G. Boccignone and M. Ferraro, Ecological Sampling of Gaze Shifts,
# %                                     IEEE Trans. SMC-B,
# %
# %
# %
# % Author
# %   Giuseppe Boccignone <Giuseppe.Boccignone(at)unimi.it>
# %
# %
# % Changes
# %   12/12/2012  First Edition
# %
# %
from utils.logger import Logger
logger = Logger(__name__)


def esHyperParamUpdate(nu, disorder, order, complexity, c_eps):
    """Dirichlet hyper-parameter update.

    Computes the new Dirichlet hyper-parameter $$\nu_{k}(t)$$
    Given the complexity $\mathcal{C}(t)$, we partition the complexity range in order to define
    $K$ possible complexity events $\{E_{\mathcal{C}(t)}=k\}_{k=1}^{K}$.
    This way the hyper-parameter update can be rewritten as the recursion.


    $$\nu_{k}(t)= \nu_k(t-1) +\left[ E_{\mathcal{C}(t)} = k \right], k=1,\cdots,K$$.

    Args:
        nu (vector): old Dirichlet Hyperparameters
        Disorder (float): disorder parameter
        Order (float): order parameter
        Compl (float): complexity parameter $\mathcal{C}(t)$
        C_EPS (float): the chaos edge

    Returns:
        nu (vector): new Dirichlet Hyperparameters

    Examples:
        >>> nu = esHyperParamUpdate(nu_old, disorder, order, compl, COMPL_EDGE)

    """

    thresh = (0.25 -  c_eps) # If beyond threshold we are in the complex domain
    reset_step = 25 # Parameter to control hysteresis

    for k in range(len(nu)):
        nu[k] = nu[k] % reset_step
        if nu[k] == 0:
            nu[k] = 1

    if complexity <= thresh:
        if (disorder < order):
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



