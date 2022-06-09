
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
but neither is it completely random.

For further information, see also [1]_ and [2]_.


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

import numpy as np

from config import (GeneralConfig, ProtoConfig)
from complexity import Complexity
from gaze_sampler import GazeSampler
from feature_map import FeatureMap
from frame_processor import FrameProcessor
from gaze_sampler import GazeSampler
from interest_points import IPSampler
from proto_parameters import ProtoParameters
from salience_map import SalienceMap
from utils.logger import Logger
from utils.plotter import Plotter
from action_selector import ActionSelector

logger = Logger(__name__)


def generate_scanpath(n_obs):

    # Initialization
    frame_sampling = FrameProcessor()
    proto_params = ProtoParameters()
    ip_sampler = IPSampler()
    complexity_evaluator = Complexity()
    if GeneralConfig.EXPERIMENT_TYPE == '3DLARK_SELFRESEMBLANCE':
        feature_map = FeatureMap()
        saliency_generator = SalienceMap()
    else:
        raise NotImplementedError("EXPERIMENT_TYPE not defined")
    gaze_sampler = GazeSampler(frame_sampling, proto_params, ip_sampler, None, 0)
    plt = Plotter()

    # Previous gaze shift direction
    dir_old = 0
    all_foa = np.array([])
    pred_foa = gaze_sampler.start_foa
    nu = gaze_sampler.start_nu

    # THE ECOLOGICAL SAMPLING CYCLE UNFOLDING IN TIME
    start = GeneralConfig.NN_IMG_START + GeneralConfig.OFFSET
    end = GeneralConfig.NN_IMG_END
    step = GeneralConfig.FRAME_STEP

    # Loop through frames
    for frame in range(start, end, step):
        logger.info(f"Processing Frame #{frame}")
        # Choosing features and salience to be used within the experiment

        # A. SAMPLING THE NATURAL HABITAT (VIDEO)
        I = frame_sampling.read_frames(frame)
        # A.1 MAKES A FOVEATED FRAME
        foveated_I = frame_sampling.foveated_imaging(pred_foa, I)
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
        # B.1.1 Evaluate complexity C(t)
        order, disorder, complexity = complexity_evaluator.compute_complexity(hist_mat, n_samples, n_bins)
        # B.2 ACTION SELECTION VIA LANDSCAPE COMPLEXITY
        action_sampler = ActionSelector(disorder, order, complexity)
        nu, z = action_sampler.select_action(nu)

        # C. SAMPLING THE GAZE SHIFT
        gaze_sampler = GazeSampler(frame_sampling, proto_params, ip_sampler, hist_mat, num_proto)
        final_foa, dir_new = gaze_sampler.sample_gaze_shift(z, pred_foa, dir_old)

        # Saving the previous foa and shift direction
        pred_foa = final_foa
        dir_old = dir_new

        if GeneralConfig.VISUALIZE_RESULTS:
            data = {
                "frame_sampling": frame_sampling,
                "feature_map": feature_map.show,
                "saliency_map": saliency_map,
                "num_proto": num_proto,
                "proto_params": proto_params,
                "circle_coords": sampled_points_coord,
                "hist_mat": ip_sampler.show_hist,
                "complexity": complexity_evaluator,
                "gaze_sampler": gaze_sampler,
            }
            # Displaying relevant steps of the process.
            plt.plot_visualization(data, frame)

        all_foa = np.vstack((all_foa, final_foa)) if all_foa.size else final_foa

    plt.save_complexity(data['complexity'])
    plt.save_foa_values(all_foa)
