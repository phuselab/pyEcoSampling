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

    #-------------------------------------------------------------------------
    # FOR VISUALIZATION AND FILE SAVING
    #-------------------------------------------------------------------------
    VERBOSE = False # Comment visualization
    VISUALIZE_RESULTS = 1

    # For saving need to set VISUALIZE_RESULTS=1
    SAVE_FOV_IMG = 0  # If = 1, saving the foveated image
    SAVE_SAL_IMG = 0  # If = 1, saving the salience map
    SAVE_PROTO_IMG = 0  # If = 1, saving the proto-object map
    SAVE_IP_IMG = 0  # If = 1, saving the interest point map
    SAVE_HISTO_IMG = 0  # If = 1, saving the histogram map
    SAVE_FOA_IMG = 0  # If = 1, saving the FOA image

    # Saving data
    SAVE_FOA_ONFILE = 0
    SAVE_COMPLEXITY_ONFILE = 0

class SaliencyConfig:
    # SELF RESEMBLANCE SPATIO-TEMPORAL FEATURE & SALIENCY MAP PARAMETERS

    # USING THE LARK PARAMETERES
    WSIZE = 3   # LARK spatial window size
    WSIZE_T = 3 # LARK temporal window size
    LARK_ALPHA = 0.42 # LARK sensitivity parameter
    LARK_H = 1  # smoothing parameter for LARK
    LARK_SIGMA = 0.7 # fall-off parameter for self-resemblamnce

    #if we perform a pyramid decomposition for efficiency
    sLevels=4; #levels of piramid decomp

class ProtoConfig:
    # PROTO-OBJECTS PARAMETERS
    PROTO = 1 #using a proto-object representation
    N_BEST_PROTO = 8 #num max of retrieved proto-object

class IPConfig:
    #-------------------------------------------------------------------------
    # INTEREST POINT SAMPLER
    #-------------------------------------------------------------------------
    # Type of interest operator to use
    INTEREST_POINT_TYPE = 'SelfResemblance'

    # Scales at which features are extracted (radius of region in pixels).
    INTEREST_POINT_SCALE = np.arange(10, 30)

    # Maximum number of interest points allowed per image
    INTEREST_POINT_MAX_POINTS = 80 #30 original

    # Parameters for particular type of detector
    INTEREST_POINT_WEIGHTED_SAMPLING = 1 #1= saliency weighted density, 0 = random sampling
    INTEREST_POINT_WEIGHTED_SCALE = 0

    WINDOW_SIZE = 7 #spatial resolution of IP: this should be set as a function of the scale at which IP has been detected

    WITH_PERTURBATION = True # Adds some other IPs by sampling directly from the salience
                            # landscape

    #-------------------------------------------------------------------------
    # HISTOGRAM OPERATOR FOR IPs EMPIRICAL DISTRIBUTION
    #-------------------------------------------------------------------------
    X_BIN_SIZE = 20
    Y_BIN_SIZE = 20


class ComplexityConfig:

    COMPLEXITY_TYPE = 'SDL';  # Shiner-Davison-Landsberg (SDL) complexity
    # complexity_type  = 'LMC'; # Lòpez-Ruiz, Mancini, and Calbet complexity
    # complexity_type  = 'FC';  # Feldman and Crutchfield?s amendment replaces D with
                                # the Kullback-Leibler divergence
    COMPL_EPS = 0.004 # 0.002


class GazeConfig:
    #-------------------------------------------------------------------------
    # GAZE SAMPLING SETTINGS
    #-------------------------------------------------------------------------

    FIRST_FOA_ON_CENTER = 1; # If == 1 sets the first Foa on frame center

    SIMPLE_ATTRACTOR = 0; # If == 1 using one point attractor in the potential
                        # otherwise using multipoints

    # NMAX number of potential FOAS to determine the total attractor potential in
    # LANGEVIN:
    #    HI...HNMAX. H=(X-X0)^2 , dH/dX=2(X-X0)
    #  the data
    NMAX = 10; # This is used if SIMPLE_ATTRACTOR=0;

    # Internal simulation: somehow related to visibility: the more the points
    # that can be sampled the higher the visibility of the field

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


