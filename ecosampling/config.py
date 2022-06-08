"""Global configuration script.

Holds all settings used in all parts of the code, enabling the exact
reproduction of the experiment at some future date.

Single most important setting - the overall experiment type
used by esGenerateScanpath.m

Authors:
    Giuseppe Boccignone <Giuseppe.Boccignone(at)unimi.it>
    Renato A Nobre <renato.avellarnobre(at)unimi.it>

Changes:
    - 20/05/2022  Python Edition
    - 20/01/2012  Matlab Edition
"""

import os
import numpy as np


class GeneralConfig:

    """Identifies Feature Extraction and Salience Map methods."""
    EXPERIMENT_TYPE = '3DLARK_SELFRESEMBLANCE'

    # General Parameters
    OFFSET = 1
    FRAME_STEP = 2  # setting the frames to skip

    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    FRAME_DIR = os.path.join(BASE_DIR, 'data/demo/')

    # % Video name:
    VIDEO_NAME = 'beverly01' # From CNRS dataset: clip beverly01
    DIR_OFFSET = 310 # First numbered frame of the directory
    # Select frames
    NN_IMG_START = 311 - DIR_OFFSET
    NN_IMG_END = 400 - DIR_OFFSET

    RESULTS_DIR = os.path.join(BASE_DIR, 'results/')

    """Flag to log verbose information."""
    VERBOSE = False

    """Flag for visualizing results on runtime."""
    VISUALIZE_RESULTS = 1

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
    PROTO = 1

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
    WEIGHTED_SAMPLING = 1

    WEIGHTED_SCALE = 0

    """Number of points used on non-weighted sampling."""
    N_POINTS = 0

    """Spatial Resolution of IPs. Should be set as a function of the sacle of IP detection."""
    WINDOW_SIZE = 7

    """Flag to sample e other IPs directly from the salience landscape."""
    WITH_PERTURBATION = True

    #-------------------------------------------------------------------------
    # HISTOGRAM OPERATOR FOR IPs EMPIRICAL DISTRIBUTION
    #-------------------------------------------------------------------------
    X_BIN_SIZE = 20
    Y_BIN_SIZE = 20


class ComplexityConfig:
    """Complexity parameters configuration."""

    """Complexity parameters. Available only 'SDL', 'LMC', 'FC'."""
    TYPE = 'SDL'

    """ """
    EPS = 0.004


class GazeConfig:
    """Gaze sampling settings."""

    """Sets the first Foa on frame center if true."""
    FIRST_FOA_ON_CENTER = True

    """Using one point attractor in the potential if true, otherwise using multipoints."""
    SIMPLE_ATTRACTOR = False

    """Number of potention FOAS to determine the total attractor portential in Langevin."""
    NMAX = 10

    # Internal simulation: somehow related to visibility: the more the points
    # that can be sampled the higher the visibility of the field of view.
    NUM_INTERNALSIM = 100; # Maximum allowed number of candidate new gaze position r_new
    # If anything goes wrong retry:
    MAX_NUMATTEMPTS = 5; # Maximum allowed tries for sampling e new valid gaze position

    # Setting the parameters of the alpha-stable components
    # -alpha is the exponent (alpha=2 for gaussian, alpha=1 for cauchian)
    # -gamma is the standard deviation
    # -beta  is symmetry parameter
    # -delta  is location parameter (for no drift, set to 0)

    # Vector pos 0 is the 'normal gaze';
    # Vector pos 1 is the 'Levy flight 1';
    # Vector  pos 2 is the 'Levy flight 2'
    ALPHA_STABLE = [2.0, 1.6, 1.4]
    GAMMA_STABLE = [3.78, 22, 60]
    BETA_STABLE = [1, 1, 1]
    DELTA_STABLE = [9, 60, 250]


