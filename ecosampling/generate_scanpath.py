
import numpy as np

from complexity import Complexity
from config import GazeConfig, GeneralConfig, ProtoConfig
from feature_map import FeatureMap
from frame_processor import FrameProcessor
from interest_points import IPSampler
from proto_parameters import ProtoParameters
from salience_map import SalienceMap
from utils.logger import Logger
from utils.plotter import Plotter

logger = Logger(__name__)

# function esGenerateScanpath(config_file,  nOBS)
# %esGenerateScanpath - Generates a scanpath on video by computing gaze shifts
# %                     through Ecological Sampling (ES)
# %
# % Synopsis
# %          esGenerateScanpath(config_file)
# %
# % Description
# %     Baseline implementation of the Ecological Sampling model, a stochastic model of eye guidance
# %     The gaze shift mechanism is conceived as  an active random sampling  that
# %     the "foraging eye" carries out upon the visual landscape,
# %     under the constraints set by the  observable features   and the
# %     global complexity of the  landscape.
# %     The actual  gaze relocation is  driven by a stochastic differential equation
# %     whose noise source is sampled from a mixture of $$\alpha$$-stable distributions.
# %     The sampling strategy  allows to mimic a fundamental property of  eye guidance:
# %     where we choose to look next at any given moment in time is not completely deterministic,
# %     but neither is it completely random
# %
# %   See the comments in each routine for details of what it does
# %   Settings for the experiment should be held in the configuration
# %   file.
# %
# % Inputs ([]s are optional)
# %   (string) config_file  the name of a configuration file in the .\config
# %                         directory to be evaluated for setting the
# %                         experiment parameters
# %
# % Outputs ([]s are optional)
# %
# %   ....
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
# % Author
# %   Giuseppe Boccignone <Giuseppe.Boccignone(at)unimi.it>
# %
# %
# % Changes
# %   12/12/2012  First Edition
# %

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

    # Setting parameters for the $$Dirichlet(\pi; nu0,nu1,nu2)$$ distribution
    nu = np.ones(3) # We start with equal probabilities

    # Setting sampling parameters
    gaze_sampling_params = {}
    # Internal simulation: somehow related to visibility: the more the points
    # that can be sampled the higher the visibility of the field
    gaze_sampling_params["NUM_INTERNALSIM"] = GazeConfig.NUM_INTERNALSIM # Maximum allowed number of candidate new  gaze position r_new
    # % If anything goes wrong retry:
    gaze_sampling_params["MAX_NUMATTEMPTS"] = GazeConfig.MAX_NUMATTEMPTS; # Maximum allowed tries for sampling e new valid gaze position

    # Setting parameters for the alpha-stable distribution
    alpha_stable_params = {}
    alpha_stable_params["alpha"] = GazeConfig.ALPHA_STABLE
    alpha_stable_params["beta"]  = GazeConfig.BETA_STABLE
    alpha_stable_params["gamma"] = GazeConfig.GAMMA_STABLE
    alpha_stable_params["delta"] = GazeConfig.DELTA_STABLE

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
        complexity_evaluator.compute_complexity(hist_mat, n_samples, n_bins)

#     %B.2. ACTION SELECTION VIA LANDSCAPE COMPLEXITY

#     % Dirichlet hyper-parameter update
#     nu = esHyperParamUpdate(nu, Disorder, Order, Compl, COMPL_EPS);
#     if VERBOSE
#         fprintf('\n Complexity  %g // Order %g // Disorder %g', Compl, Order, Disorder);
#         fprintf('\n Parameter nu1 %g', nu(1));
#         fprintf('\n Parameter nu2 %g', nu(2));
#         fprintf('\n Parameter nu3 %g', nu(3));
#     end

#     % Sampling the \pi parameter that is the probability of an order event
#     %   $$\pi ~ %Dir(\pi | \nu)$$
#     pi_prob = dirichlet_sample(nu, 1);

#     % Sampling the kind of gaze-shift regime:
#     %   $$ z ~ Mult(z | \pi) $$
#     z = sample_discrete(pi_prob,1,1);

#     if VERBOSE
#         fprintf('\n Action sampled: z = %d \n', z);
#     end

#     %C. SAMPLING THE GAZE SHIFT
#     %
#     if VERBOSE
#         fprintf('\n Sample gaze point \n');

#     end

#     predFOA = finalFOA; % saving the previous FOA
#     if mod(n,1)==0
#         %setting the landscape
#         if numproto
#            landscape.areaProto = areaProto;
#            landscape.protObject_centers = protObject_centers;
#         else
#            landscape.histmat  = histmat;
#            landscape.xbinsize = xbinsize;
#            landscape.ybinsize = ybinsize;
#            landscape.NMAX     = NMAX;
#         end

#         %setting the attractors of the FOA
#         FOA_attractors = esGetGazeAttractors(landscape, predFOA, numproto, SIMPLE_ATTRACTOR);

#         %sampling the FOA, which is returned togethre with the simulated candidates
#         % setting the oculomotor state z parameter;
#         gazeSampParam.z = z;
#         [finalFOA  dir_new  candx candy candidateFOA] = esGazeSampling(gazeSampParam, foaSize, FOA_attractors,nrow,ncol,...
#                                                                         predFOA, dir_old, alpha_stblParam,xCord, yCord);

#         dir_old = dir_new; % saving the previous shift directio

#         % if set, save the FOA coordinates on file
#         if SAVE_FOA_ONFILE
#          allFOA= [allFOA ;finalFOA];
#         end
#     end

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


#         % 9. The sampled FOA
#         countpics=countpics+1;
#         subplot(NUMLINE, NUMPICSLINE, countpics);
#         %title('Final FOA'), hold on ;
#         %making FOA
#         rad=foaSize; BW = mkDisc(size(currFrame), rad, finalFOA');
#         BW=logical(BW);   BW2= BW; BW2=logical(1-BW);
#         rgb = imoverlay(currFrame, BW2, [0 0 0]);
#         sc(rgb); label(rgb, 'Final FOA');
#         if SAVE_FOA_IMG
#             [X,MAP]= frame2im(getframe);
#             FILENAME=[RESULT_DIR VIDEO_NAME '/FOA/FOA' imglist(iFrame).name];
#             imwrite(X,FILENAME,'jpeg');
#         end
#     drawnow
#     end %VISUALIZE

# end % ES gaze shift loop

# % Save some results if configured
# if SAVE_COMPLEXITY_ONFILE
#     outfilenameord=[RESULT_DIR VIDEO_NAME 'order.mat']
#     save(outfilenameord,'Orderplot');
#     outfilenameord=[RESULT_DIR VIDEO_NAME 'disorder.mat']
#     save(outfilenameord,'Disorderplot');
# end

# if SAVE_FOA_ONFILE
#     outfilename=[RESULT_DIR VIDEO_NAME '_FOA.mat']
#     save(outfilename,'allFOA');
# end

# end % FUNCTION

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

    for n_obs in range(total_observers):
       #  Generate and visualize an ES scanpath
       #  Calling the overall routine esGenerateScanpath that does everything, with
       #  a configuration file: the routine will run each subsection of the gaze shift
       #  scheme in turn.
       esGenerateScanpath(configFileName,  n_obs)
