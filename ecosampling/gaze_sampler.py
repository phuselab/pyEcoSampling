
import numpy as np
from config import GazeConfig

class GazeSampler:

    def __init__(self):
        # Setting parameters for the $$Dirichlet(\pi; nu0,nu1,nu2)$$ distribution
        self.nu = np.ones(3) # We start with equal probabilities
        # Setting sampling parameters
        self.params = {
            # Maximum allowed number of candidate new  gaze position r_new
            "NUM_INTERNALSIM": GazeConfig.NUM_INTERNALSIM,
            # Maximum allowed tries for sampling e new valid gaze position
            "MAX_NUMATTEMPTS": GazeConfig.MAX_NUMATTEMPTS,
        }
        # Setting parameters for the alpha-stable distribution
        self.alpha_stable_params = {
            "alpha": GazeConfig.ALPHA_STABLE,
            "beta": GazeConfig.BETA_STABLE,
            "gamma": GazeConfig.GAMMA_STABLE,
            "delta": GazeConfig.DELTA_STABLE
        }
        # Internal simulation: somehow related to visibility: the more the points
        # that can be sampled the higher the visibility of the field



