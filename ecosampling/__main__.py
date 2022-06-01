"""Top level script that runs a Ecological Sampling (ES) experiment.

The experiment consists in putting into action a defined number of
artificial observers, each generating a visual scanpath
on a given video. All paremeters defining the experiment are
defined in the config.py script file

Authors:
    Giuseppe Boccignone <giuseppe.boccignone@unimi.it>
    Renato Nobre <renato.avellarnobre@studenti.unimi.it>

Changes:
    12/12/2012  First Edition Matlab
    31/05/2022  Python Edition

References
----------
.. [1] `G. Boccignone and M. Ferraro, Ecological Sampling of Gaze Shifts
IEEE Trans. Systems Man Cybernetics - Part B (on line IEEExplore).
< >`_
.. [2] `G. Boccignone and M. Ferraro, The active sampling of gaze-shifts,
in Image Analysis and Processing ICIAP 2011, ser. Lecture Notes in Computer Science,
G. Maino and G. Foresti, Eds. Springer Berlin / Heidelberg, 2011, vol. 6978, pp. 187?196.
<>`_
"""

from generate_scanpath import esGenerateScanpath


# See also
#   esGenerateScanpath
#   config_<type of experiment>

if __name__ == "__main__":
    # Set here the total number of observers / scanpaths to be simulated
    total_observers = 1
    # Set the configuration filename (parameters) of the experiment
    configFileName = 'config_demo'

    for n_obs in range(total_observers):
       esGenerateScanpath(configFileName,  n_obs)
