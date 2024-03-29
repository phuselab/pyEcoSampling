"""Top level script that runs a Ecological Sampling (ES) experiment.

The experiment consists in putting into action a defined number of
artificial observers, each generating a visual scanpath
on a given video. All paremeters defining the experiment are
defined in the config.py script file

Authors:
    - Giuseppe Boccignone <giuseppe.boccignone@unimi.it>
    - Renato Nobre <renato.avellarnobre@studenti.unimi.it>

Changes:
    - 12/12/2012  First Edition Matlab
    - 31/05/2022  Python Edition

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

from config import SaliencyConfig, GeneralConfig
from generate_scanpath import generate_scanpath

from utils.logger import Logger

logger = Logger(__name__)

if __name__ == "__main__":

    if SaliencyConfig.WSIZE != 3 or SaliencyConfig.WSIZE_T != 3:
        raise NotImplementedError("WSIZE and WSIZE_T must be 3")

    for n_obs in range(GeneralConfig.TOTAL_OBSERVERS):
        logger.info(f"Generating scanpath for observer {n_obs+1}")
        generate_scanpath(n_obs)
