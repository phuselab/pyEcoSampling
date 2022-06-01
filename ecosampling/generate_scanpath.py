
"""Generates a scanpath on video by computing gaze shifts through Ecological Sampling (ES).

Baseline implementation of the Ecological Sampling model, a stochastic model of eye guidance
The gaze shift mechanism is conceived as an active random sampling that
the "foraging eye" carries out upon the visual landscape,
under the constraints set by the observable features and the
global complexity of the  landscape.
The actual gaze relocation is driven by a stochastic differential equation
whose noise source is sampled from a mixture of alpha-stable distributions.
The sampling strategy allows to mimic a fundamental property of eye guidance:
where we choose to look next at any given moment in time is not completely deterministic,
but neither is it completely random


Notes:
    - See the comments in each routine for details of what it does
    - Settings for the experiment should be held in the
      configuration file.

Authors:
    - Giuseppe Boccignone <giuseppe.boccignone@unimi.it>
    - Renato Nobre <renato.avellarnobre@studenti.unimi.it>

Changes:
    - 12/12/2012  First Edition Matlab
    - 31/05/2022  Python Edition
"""

import numpy as np
import pymc3 as pm

from complexity import Complexity
from config import (ComplexityConfig, GazeConfig, GeneralConfig, IPConfig,
                    ProtoConfig)
# from gaze_sampler import GazeSampler
# from esGazeSampling import esGazeSampling
# from esGetGazeAttractors import esGetGazeAttractors
# from esHyperParamUpdate import esHyperParamUpdate
from feature_map import FeatureMap
from frame_processor import FrameProcessor
from gaze_sampler import GazeSampler
from interest_points import IPSampler
from proto_parameters import ProtoParameters
from salience_map import SalienceMap
from utils.logger import Logger
from utils.plotter import Plotter

logger = Logger(__name__)


# % Inputs ([]s are optional)
# %   (string) config_file  the name of a configuration file in the .\config
# %                         directory to be evaluated for setting the
# %                         experiment parameters
# %
# % Example:
# %
# %   esGenerateScanpath('config');
# %

# % References
# %   [1] G. Boccignone and M. Ferraro, Ecological Sampling of Gaze Shifts
# %       IEEE Trans. Systems Man Cybernetics - Part B (to appear)
# %   [2] G. Boccignone and M. Ferraro, The active sampling of gaze-shifts,
# %       in Image Analysis and Processing ICIAP 2011,
# %       ser. Lecture Notes in Computer Science,
# %       G. Maino and G. Foresti, Eds.	Springer Berlin / Heidelberg, 2011,
# %       vol. 6978, pp. 187?196.


def esGenerateScanpath(config_file,  n_obs):

    ## INITIALIZATION
    # The EXPERIMENT_TYPE sets Feature Extraction and Salience Map methods
    # Setting feature parameters
    if GeneralConfig.EXPERIMENT_TYPE == '3DLARK_SELFRESEMBLANCE':
        # Set 3-D LARKs Parameters
        feature_map = FeatureMap()
        saliency_generator = SalienceMap()
    else:
        raise NotImplementedError("EXPERIMENT_TYPE not defined")


    # Set proto object structure
    proto_params = ProtoParameters()
    gaze_sampler = GazeSampler()
    complexity_evaluator = Complexity()
    frame_sampling = FrameProcessor()

    if GeneralConfig.SAVE_FOA_ONFILE:
        all_FOA = []

    # Set first FOA: default frame center
    foa_size = round(max(frame_sampling.n_rows, frame_sampling.n_cols) / 6)
    if GazeConfig.FIRST_FOA_ON_CENTER:
        x_center = round(frame_sampling.n_rows/2)
        y_center = round(frame_sampling.n_cols/2)
    else:
        print('\n No methods defined for setting the first FOA')

    final_foa = np.array([x_center, y_center])
    nu = np.ones(3)
    # Number of iterations
    n = 0
    # Previous gaze shift direction
    dir_old = 0

    plt = Plotter()

    # THE ECOLOGICAL SAMPLING CYCLE UNFOLDING IN TIME
    start = GeneralConfig.NN_IMG_START + GeneralConfig.OFFSET
    end = GeneralConfig.NN_IMG_END
    step = GeneralConfig.FRAME_STEP
    for frame in range(start, end, step):
        logger.info(f"Processing Frame #{frame}")

        # Choosing features and salience to be used within the experiment

        n += 1
        # A. SAMPLING THE NATURAL HABITAT (VIDEO)
        I = frame_sampling.read_frames(frame)
        # A.1 MAKES A FOVEATED FRAME
        foveated_I = frame_sampling.foveated_imaging(final_foa, I)
        # A.2 COMPUTE FEATURES OF THE PHYSICAL LANDSCAPE
        reduced_frames = frame_sampling.reduce_frames(foveated_I)
        # The method for computing features has been set in the config_file
        foveated_feature_map = feature_map.compute_features(reduced_frames, frame_sampling)
        # A.3 SAMPLE THE FOVEATED SALIENCE MAP
        saliency_map = saliency_generator.compute_salience(foveated_feature_map, frame_sampling)

        num_proto = 0
        if ProtoConfig.PROTO:
            # A.4 SAMPLE PROTO-OBJECTS
            num_proto = proto_params.sample_proto_objects(saliency_map)
            # A.5 SAMPLE INTEREST POINTS / PREYS
            ip_sampler = IPSampler()
            sampled_points_coord = ip_sampler.interest_point_sample(num_proto, proto_params, saliency_map)

        # B. SAMPLING THE APPROPRIATE OCULOMOTOR ACTION
        # B.1 EVALUATING THE COMPLEXITY OF THE SCENE
        hist_mat, n_samples, n_bins = ip_sampler.histogram_ips(frame_sampling, sampled_points_coord)
        # Step 2 Evaluate complexity $$C(t)$$
        order, disorder, complexity = complexity_evaluator.compute_complexity(hist_mat, n_samples, n_bins)

        # B.2. ACTION SELECTION VIA LANDSCAPE COMPLEXITY
        # Dirichlet hyper-parameter update
        nu = esHyperParamUpdate(nu, disorder, order, complexity, ComplexityConfig.EPS)
        logger.verbose(f"Complexity  {complexity} // Order {order} // Disorder {disorder}")
        logger.verbose(f"Parameter nu1 {nu[0]}")
        logger.verbose(f"Parameter nu2 {nu[1]}")
        logger.verbose(f"Parameter nu3 {nu[2]}")

        # Sampling the \pi parameter that is the probability of an order event
        # $$\pi ~ %Dir(\pi | \nu)$$
        dirchlet_dist = pm.Dirichlet.dist(nu)
        pi_prob = dirchlet_dist.random(size=1)

        # Sampling the kind of gaze-shift regime:
        # $$ z ~ Mult(z | \pi) $$
        # z = sample_discrete(pi_prob,1,1)

        # logger.verbose(f"Action sampled: z = {z}")

        # # C. SAMPLING THE GAZE SHIFT
        # logger.verbose("Sample gaze point")

        # pred_foa = final_foa # Saving the previous FOA
        # if n % 1 == 0:
        #     # Setting the landscape
        #     landscape = {}
        #     if num_proto > 0:
        #        landscape["area_proto"] = proto_params.area_proto
        #        landscape["proto_centers"] = proto_params.proto_centers
        #     else:
        #        landscape["histmat"] = hist_mat
        #        landscape["xbinsize"] = IPConfig.X_BIN_SIZE
        #        landscape["ybinsize"] = IPConfig.Y_BIN_SIZE
        #        landscape["NMAX"] = GazeConfig.NMAX

        # # Setting the attractors of the FOA
        # foa_attractors = esGetGazeAttractors(landscape, pred_foa, num_proto)

        # # sampling the FOA, which is returned togethre with the simulated candidates
        # # setting the oculomotor state z parameter;
        # gaze_sampling_params["z"] = z
        # final_foa, dir_new, candx, candy, candidateFOA = esGazeSampling(gaze_sampling_params,
        #                                                                foa_size, foa_attractors,
        #                                                                nrow, ncol, predFOA, dir_old,
        #                                                                alpha_stblParam,xCord, yCord)

        # dir_old = dir_new; # saving the previous shift directio

        # if set, save the FOA coordinates on file
        # if SAVE_FOA_ONFILE
        #     allFOA.append([allFOA ;finalFOA];


        if GeneralConfig.VISUALIZE_RESULTS:

            data = {
                "original_frame": frame_sampling.current_frame,
                "foveated_frame": frame_sampling.current_foveated_frame,
                "feature_map": feature_map.show,
                "saliency_map": saliency_map,
                "proto_mask": proto_params.show_proto,
                "num_proto": num_proto,
                "proto_params": proto_params.a,
                "nV": proto_params.nV,
                "circle_coords": sampled_points_coord,
                "hist_mat": ip_sampler.show_hist,
                "order": complexity_evaluator.order,
                "disorder": complexity_evaluator.disorder,
                "complexity": complexity_evaluator.complexity,
            }
            # Displaying relevant steps of the process.
            plt.plot_visualization(data)

# % Save some results if configured
# if SAVE_COMPLEXITY_ONFILE
#     outfilenameord=[RESULT_DIR VIDEO_NAME 'order.mat']
#     save(outfilenameord,'Orderplot');
#     outfilenameord=[RESULT_DIR VIDEO_NAME 'disorder.mat']
#     save(outfilenameord,'Disorderplot');

# if SAVE_FOA_ONFILE
#     outfilename=[RESULT_DIR VIDEO_NAME '_FOA.mat']
#     save(outfilename,'allFOA');
