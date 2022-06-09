"""Global configuration script.

Holds all settings used in all parts of the code, enabling the exact
reproduction of the experiment at some future date.

Single most important setting - the overall experiment type
used by generate_scanpath.m

Authors:
    - Giuseppe Boccignone <Giuseppe.Boccignone(at)unimi.it>
    - Renato A Nobre <renato.avellarnobre(at)unimi.it>

Changes:
    - 20/05/2022  Python Edition
    - 20/01/2012  Matlab Edition
"""

import os
import numpy as np


class GeneralConfig:
    """General configuration class."""

    """Identifies Feature Extraction and Salience Map methods."""
    EXPERIMENT_TYPE = '3DLARK_SELFRESEMBLANCE'

    """Start offset from the data folder. Skip ``OFFSET`` first images."""
    OFFSET = 1

    """Number of times to execute the scanpath experiment."""
    TOTAL_OBSERVERS = 1

    FRAME_STEP = 2
    """Frames to skip at each step.

    Note:
        Instead of looping through all the frames, the step is two because
        at each step, we process the previous and the current frame and future
        frames.
    """

    """Name of the video to be processed. Must match your data frame names."""
    VIDEO_NAME = 'beverly01'

    """Number of the first frame in the directory."""
    DIR_OFFSET = 310

    """Name of the experiment folder inside data."""
    EXPERIMENT_DATA_FOLDER = 'demo/'

    """Select the start frames to be used in the experiment."""
    NN_IMG_START = 311 - DIR_OFFSET

    """Select the end frames to be used in the experiment."""
    NN_IMG_END = 400 - DIR_OFFSET

    """Flag to log verbose information."""
    VERBOSE = False

    """Flag for visualizing results on runtime."""
    VISUALIZE_RESULTS = True

    """Flag for saving foveated images."""
    SAVE_FOV_IMG = True

    """Flag for saving saliency maps."""
    SAVE_SAL_IMG = True

    """Flag for saving the proto-objects maps."""
    SAVE_PROTO_IMG = True

    """Flag for saving the interest point map."""
    SAVE_IP_IMG = True

    """Flag for saving the 2d histogram."""
    SAVE_HISTO_IMG = True

    """Flag for saving the foa images."""
    SAVE_FOA_IMG = True

    """Flag for saving coordinates of FOA on file."""
    SAVE_FOA_ONFILE = True

    """Flag for saving complexity values and plots."""
    SAVE_COMPLEXITY_ONFILE = True

    """Project root folder. Don't change this."""
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    """Project base folder. Don't change this."""
    BASE_DIR = os.path.dirname(PROJECT_ROOT)

    """Data dir folder. Don't change this."""
    DATA_DIR = os.path.join(BASE_DIR, 'data/')

    """Frame dir folder. Don't change this."""
    FRAME_DIR = os.path.join(DATA_DIR, EXPERIMENT_DATA_FOLDER)

    """Name of the folder where the results will be saved."""
    RESULTS_DIR = os.path.join(BASE_DIR, 'results/' + VIDEO_NAME + '/')

class SaliencyConfig:
    """Self resemblance spatio-temporal feature and saliency map parameters."""

    """LARK Spatial Window Size."""
    WSIZE = 3

    """LARK Temporal Window Size."""
    WSIZE_T = 3

    """LARK Sensitivity Parameter."""
    LARK_ALPHA = 0.42

    """LARK Smoothing Parameter."""
    LARK_H = 1

    """Lark fall-off parameter for self-resemblance."""
    LARK_SIGMA = 0.7

    """Levels of the pyramid decomposition (if we perform such)."""
    S_LEVELS = 4

class ProtoConfig:
    """Proto-object parameters."""

    """Using a proto-object representation."""
    PROTO = True

    """Maximum number of proto-objects."""
    N_BEST_PROTO = 15

class IPConfig:
    """Interest point sampler configuration."""

    """Type of interest operator to use."""
    TYPE = 'SelfResemblance'

    """Scales at which features are extracted (radius of region in pixels)."""
    SCALE = np.arange(10, 30)

    """Maximun number of IPs allowed per image."""
    MAX_POINTS = 80

    """If true perform weighted density, false perform random sampling"""
    WEIGHTED_SAMPLING = True

    """Scale in case of weighted sampling."""
    WEIGHTED_SCALE = 0

    """Number of points used on non-weighted sampling."""
    N_POINTS = 0

    """Spatial Resolution of IPs. Should be set as a function of the scale of IP detection."""
    WINDOW_SIZE = 7

    """Flag to sample e other IPs directly from the salience landscape."""
    WITH_PERTURBATION = True

    """Number of X bins for the IPs Empirical Distribution 2D histogram."""
    X_BIN_SIZE = 20

    """Number of Y bins for the IPs Empirical Distribution 2D histogram."""
    Y_BIN_SIZE = 20


class ComplexityConfig:
    """Complexity parameters configuration."""

    """Complexity parameters. Available only 'SDL', 'LMC', 'FC'."""
    TYPE = 'SDL'

    EPS = 0.004
    """Simulated epsilon of the machine.

    Note:
        Epsilon is the minimum distance that a floating point arithmetic
        program can recognize between two numbers x and y.
    """


class GazeConfig:
    """Gaze sampling settings.

    For the alpha-stable distribution parameters, the following
    indexes represtent the following parameters:

        - Position 0 corresponds to Normal Gaze
        - Position 1 corresponds to Levy Flight 1
        - Position 2 corresponds to Levy Flight 2
    """

    """Sets the first Foa on frame center if true."""
    FIRST_FOA_ON_CENTER = True

    """Using one point attractor in the potential if true, otherwise using multipoints."""
    SIMPLE_ATTRACTOR =  False

    """Number of potention FOAS to determine the total attractor portential in Langevin."""
    NMAX = 10

    """Maximum number of candidates new gaze positions"""
    NUM_INTERNALSIM = 100

    """Number of retries to find a valid new gaze position."""
    MAX_NUMATTEMPTS = 5

    """Possible exponents of the alpha-stable distribution."""
    ALPHA_STABLE = [2.0, 1.6, 1.4]

    """Possible standard deviation of the alpha-stable distribution."""
    GAMMA_STABLE = [3.78, 22, 60]

    """Possible symmetry of the alpha-stable distribution."""
    BETA_STABLE = [1, 1, 1]

    """Possible locations of the alpha-stable distribution."""
    DELTA_STABLE = [9, 60, 250]


