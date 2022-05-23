from esGenerateScanpath import esGenerateScanpath
# runEcologicalSampling-  Top level script that runs a
#                    Ecological Sampling (ES) experiment.
#                    The experiment consists in putting into action a
#                    defined number of
#                    artificial observers, each generating a visual scanpath
#                    on a given
#                    video
#                    All paremeters defining the experiment are
#                    defined in the config_<type of experiment>.m script
#                    file
#
# See also
#   esGenerateScanpath
#   config_<type of experiment>
#
# Requirements
#   Image Processing toolbox
#   Statistical toolbox

# References
#   [1] G. Boccignone and M. Ferraro, Ecological Sampling of Gaze Shifts
#       IEEE Trans. Systems Man Cybernetics - Part B (on line IEEExplore)
#
#   [2] G. Boccignone and M. Ferraro, The active sampling of gaze-shifts,
#       in Image Analysis and Processing ICIAP 2011,
#       ser. Lecture Notes in Computer Science,
#       G. Maino and G. Foresti, Eds.	Springer Berlin / Heidelberg, 2011,
#       vol. 6978, pp. 187?196.
#
# Authors
#   Giuseppe Boccignone <Giuseppe.Boccignone(at)unimi.it>
#
# License
#   The program is free for non-commercial academic use. Please
#   contact the authors if you are interested in using the software
#   for commercial purposes. The software must not modified or
#   re-distributed without prior permission of the authors.
#
# Changes
#   20/01/2012  First Edition


if __name__ == "__main__":
    # Set here the total number of observers / scanpaths to be simulated
    total_observers = 1

    # Set the configuration filename (parameters) of the experiment
    configFileName = 'config_demo'

    for n_obs in range(1, total_observers+1):
       #  Generate and visualize an ES scanpath
       #  Calling the overall routine esGenerateScanpath that does everything, with
       #  a configuration file: the routine will run each subsection of the gaze shift
       #  scheme in turn.
       esGenerateScanpath(configFileName,  n_obs);
