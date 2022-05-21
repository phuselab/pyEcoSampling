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


"""Identifies Feature Extraction and Salience Map methods."""
EXPERIMENT_TYPE = '3DLARK_SELFRESEMBLANCE';



# %-------------------------------------------------------------------------
# % DIRECTORIES - please change to your needs
# %-------------------------------------------------------------------------
# % Directory holding the code
# runpath = pwd;

# RUN_DIR = [ runpath '/' ]

# % Directory holding all the source frame
# FRAME_DIR = [ RUN_DIR 'Datasets/demo/']

# % Video name:
# VIDEO_NAME   = 'beverly01' %from CNRS dataset: clip beverly01
# dir_offset   = 310; %first numbered frame of the directory
# nNImageStart = 312-dir_offset; nNImageEnd=400-dir_offset;%select frames

# % Directory holding all the experiment results
# RESULT_DIR = [RUN_DIR 'results/'];




# General Parameters
offset=1
frameStep=2  # setting the frames to skip


# SELF RESEMBLANCE SPATIO-TEMPORAL FEATURE & SALIENCY MAP PARAMETERS

# USING THE LARK PARAMETERES
WSIZE = 3   # LARK spatial window size
WSIZE_T = 3 # LARK temporal window size
LARK_ALPHA = 0.42 # LARK sensitivity parameter
LARK_H = 1  # smoothing parameter for LARK
LARK_SIGMA = 0.7 # fall-off parameter for self-resemblamnce

#if we perform a pyramid decomposition for efficiency
sLevels=4; #levels of piramid decomp


# PROTO-OBJECTS PARAMETERS

PROTO      = 1 #using a proto-object representation
nBestProto = 15 #num max of retrieved proto-object


# %-------------------------------------------------------------------------
# % INTEREST POINT SAMPLER
# %-------------------------------------------------------------------------
# % Type of interest operator to use
# Interest_Point.Type                 = 'SelfResemblance';

# % Scales at which features are extracted (radius of region in pixels).
# Interest_Point.Scale                = [10:30];

# % Maximum number of interest points allowed per image
# Interest_Point.Max_Points           = 80; %30 original

# % Parameters for particular type of detector
# Interest_Point.Weighted_Sampling    = 1; %1= saliency weighted density, 0 = random sampling
# Interest_Point.Weighted_Scale = 0;

# windowSize=7; %spatial resolution of IP: this should be set as a function of the scale at which IP has been detected

# WITH_PERTURBATION = true; % Adds some other IPs by sampling directly from the salience
#                           % landscape

# %-------------------------------------------------------------------------
# % HISTOGRAM OPERATOR FOR IPs EMPIRICAL DISTRIBUTION
# %-------------------------------------------------------------------------
# xbinsize=20 ; ybinsize=20;

# %-------------------------------------------------------------------------
# % COMPLEXITY
# %-------------------------------------------------------------------------
# complexity_type  ='SDL';   %Shiner-Davison-Landsberg (SDL) complexity
# % complexity_type  ='LMC'; %Lòpez-Ruiz, Mancini, and Calbet complexity
# % complexity_type  ='FC';  %Feldman and Crutchfield?s amendment replaces D with the Kullback-Leibler divergence


# COMPL_EPS = 0.004;% 0.002


# %-------------------------------------------------------------------------
# % GAZE SAMPLING SETTINGS
# %-------------------------------------------------------------------------

# firstFOAonCENTER = 1; % if == 1 sets the first Foa on frame center

# SIMPLE_ATTRACTOR = 0; % if == 1 using one point attractor in the potential
#                       % otherwise using multipoints

# %NMAX number of potential FOAS to determine the total attractor potential in
# %LANGEVIN:
# %   HI...HNMAX. H=(X-X0)^2 , dH/dX=2(X-X0)
# % the data
# NMAX=10; %This is used if SIMPLE_ATTRACTOR=0;

# % Internal simulation: somehow related to visibility: the more the points
# % that can be sampled the higher the visibility of the field

# NUM_INTERNALSIM = 100; %maximum allowed number of candidate new  gaze position r_new
# % If anything goes wrong retry:
# MAX_NUMATTEMPTS = 5; %maximum allowed tries for sampling e new valid gaze position


# %Setting the parameters of the alpha-stable components
# % -alpha is the exponent (alpha=2 for gaussian, alpha=1 for cauchian)
# % -sigma is the standard deviation
# % -beta  is symmetry parameter
# % -delta  is location parameter (for no drift, set to 0)

# %1. 'Normal gaze'
# alpha_stable(1) = 2.0; gamma_stable(1) = 3.78; beta_stable(1) = 1;
# delta_stable(1) = 9;

# %2. 'Levy flight 1'
# alpha_stable(2) = 1.6; gamma_stable(2)= 22; beta_stable(2) = 1;
# delta_stable(2) = 60;

# %3. 'Levy flight 2'
# alpha_stable(3) = 1.4; gamma_stable(3) = 60; beta_stable(3) = 1;
# delta_stable(3) = 250;


# %-------------------------------------------------------------------------
# % FOR VISUALIZATION AND FILE SAVING
# %-------------------------------------------------------------------------
# VERBOSE = false; %comment visualization

# VISUALIZE_RESULTS = 1;

# %for saving need to set VISUALIZE_RESULTS=1
# SAVE_FOV_IMG   = 0;  %if = 1, saving the foveated image
# SAVE_SAL_IMG   = 0;  %if = 1, saving the salience map
# SAVE_PROTO_IMG = 0;  %if = 1, saving the proto-object map
# SAVE_IP_IMG    = 0;  %if = 1, saving the interest point map
# SAVE_HISTO_IMG = 0;
# SAVE_FOA_IMG   = 0;  %if = 1, saving the FOA image

# %saving data
# SAVE_FOA_ONFILE        = 0;
# SAVE_COMPLEXITY_ONFILE = 0;


