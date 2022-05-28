from turtle import shape
from config import GeneralConfig, SaliencyConfig, GazeConfig, ProtoConfig, IPConfig

import os
import cv2
import pymc3 as pm
import numpy as np

from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize

from utils.mkGaussian import mkGaussian
from esComputeFeatures import esComputeFeatures
from esComputeSaliency import esComputeSalience
from esSampleProtoMap import esSampleProtoMap
from esSampleProtoParameters import esSampleProtoParameters
from esInterestPointSampling import interest_point_sampling
from esComputeComplexity import compute_complexity

from matplotlib.patches import Ellipse, Circle



import matplotlib.pyplot as plt

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
# %
# % Requirements
# %   Image Processing toolbox
# %   Statistical toolbox

# % References
# %   [1] G. Boccignone and M. Ferraro, Ecological Sampling of Gaze Shifts
# %       IEEE Trans. Systems Man Cybernetics - Part B (to appear)
# %   [2] G. Boccignone and M. Ferraro, The active sampling of gaze-shifts,
# %       in Image Analysis and Processing ICIAP 2011,
# %       ser. Lecture Notes in Computer Science,
# %       G. Maino and G. Foresti, Eds.	Springer Berlin / Heidelberg, 2011,
# %       vol. 6978, pp. 187?196.
# %
# %
# %
# % Author
# %   Giuseppe Boccignone <Giuseppe.Boccignone(at)unimi.it>
# %
# % License
# %   The program is free for non-commercial academic use. Please
# %   contact the authors if you are interested in using the software
# %   for commercial purposes. The software must not modified or
# %   re-distributed without prior permission of the authors.
# %
# % Changes
# %   12/12/2012  First Edition
# %

def esGenerateScanpath(config_file,  n_obs):

# error(nargchk(1, 2, nargin));
# if ~exist('nOBS', 'var') || isempty(nOBS)
#     nOBS = 1;
# end


    ## INITIALIZATION
    # The EXPERIMENT_TYPE sets Feature Extraction and Salience Map methods
    # Setting feature parameters
    if GeneralConfig.EXPERIMENT_TYPE == '3DLARK_SELFRESEMBLANCE':
        # Set 3-D LARKs Parameters
        feature_type = GeneralConfig.EXPERIMENT_TYPE
        feature_params = {}
        feature_params["wsize"] = SaliencyConfig.WSIZE # LARK spatial window size
        feature_params["wsize_t"] = SaliencyConfig.WSIZE_T # LARK temporal window size
        feature_params["alpha"] = SaliencyConfig.LARK_ALPHA # LARK sensitivity parameter
        feature_params["h"] = SaliencyConfig.LARK_H  # Smoothing parameter for LARK
        feature_params["sigma"] = SaliencyConfig.LARK_SIGMA # fall-off parameter for self-resemblamnce
    else:
        print('ERROR: EXPERIMENT_TYPE not defined')

    # Setting salience parameters
    if GeneralConfig.EXPERIMENT_TYPE == '3DLARK_SELFRESEMBLANCE':
        # Set Self Resemblance Parameters
        sal_type = GeneralConfig.EXPERIMENT_TYPE;
        salience_params = {}
        salience_params["wsize"] = SaliencyConfig.WSIZE; # LARK spatial window size
        salience_params["wsize_t"] = SaliencyConfig.WSIZE_T; # LARK temporal window size
        salience_params["sigma"] = SaliencyConfig.LARK_SIGMA; # Fall-off parameter for self-resemblamnce
    else:
        print("Error!")

    # Set proto object structure
    old_protoParam = {}
    old_protoParam["B"] = []
    old_protoParam["a"] = []
    old_protoParam["r1"] = []
    old_protoParam["r2"] = []
    old_protoParam["cx"] = []
    old_protoParam["cy"] = []
    old_protoParam["theta"] = []

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

    # Allocating some vectors to be used later
    disorder_plot = []
    order_plot = []
    complexity_plot = []
    if GeneralConfig.SAVE_FOA_ONFILE:
        all_FOA = []

    # Reads the first frame and gets some parameters: frame size, etc...
    img_list = [os.path.join(GeneralConfig.FRAME_DIR, image) for image in os.listdir(GeneralConfig.FRAME_DIR) if image.endswith(".jpg")]
    img_list.sort()
    fnum = len(img_list)
    img_1 = io.imread(img_list[0])
    n_row, n_col = img_1[:,:,0].shape # Rows, columns

    # Set 2D histogram structure
    # nbinsx=fix(nrow/xbinsize); nbinsy=fix(ncol/ybinsize);
    # XLO = 1; XHI = nrow; YLO = 1; YHI =ncol;

    # Set first FOA: default frame center
    foa_size = round(max(n_row, n_col) / 6)
    if GazeConfig.FIRST_FOA_ON_CENTER:
        x_center = round(n_row/2)
        y_center = round(n_col/2)
    else:
        print('\n No methods defined for setting the first FOA')

    final_foa = np.array([x_center, y_center])

    # Set visualization parameters
    if GeneralConfig.VISUALIZE_RESULTS:
        font_size = 18
        line_width = 2
        marker_size = 16
        NUMLINE = 2
        NUM_PICS_LINE = 5
        # % Set up display window
        # scrsz = get(0,'ScreenSize');
        # figure('Position',[1 scrsz(4) scrsz(3) scrsz(4)],'Name','Ecological Sampling of Gaze Shift Demo','NumberTitle','off')

    # Number of iterations
    n = 0
    # Number of proto_objects
    numproto = 0
    # Previous gaze shift direction
    dir_old = 0

    disorder_plot = []
    order_plot = []
    complexity_plot = []

    fig, ax = plt.subplots(NUMLINE, NUM_PICS_LINE, figsize=(15, 5))
    plt.ion()
    plt.show()

    # THE ECOLOGICAL SAMPLING CYCLE UNFOLDING IN TIME
    start = GeneralConfig.NN_IMG_START + GeneralConfig.OFFSET
    end = GeneralConfig.NN_IMG_END
    step = GeneralConfig.FRAME_STEP
    for frame in range(start, end, step):
        print(f"Processing Frame #{frame}\n");

#     % Choosing features and salience to be used within the experiment
        if GeneralConfig.EXPERIMENT_TYPE == '3DLARK_SELFRESEMBLANCE':
            n_frames = SaliencyConfig.WSIZE_T
            n += 1
            # A. SAMPLING THE NATURAL HABITAT (VIDEO)
            # Reading three consecutive frames
            if GeneralConfig.VERBOSE:
                print('Data acquisition')

            pred_frame = io.imread(img_list[frame-GeneralConfig.OFFSET]) # Get previous frame
            curr_frame = io.imread(img_list[frame]) # Get current frame
            next_frame = io.imread(img_list[frame+GeneralConfig.OFFSET])# Get subsequent frame

            # Converting to grey level
            I = np.zeros((n_row, n_col, n_frames))
            I[:,:,0] = rgb2gray(pred_frame)
            # cv2.cvtColor(pred_frame, cv2.COLOR_BGR2GRAY)
            I[:,:,1] = rgb2gray(curr_frame)
            #  cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            I[:,:,2] = rgb2gray(next_frame)
            #  cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)


            # For visualization
            show_curr_frame = curr_frame

            # A.1 MAKES A FOVEATED FRAME

            #   Using:
            #   IM = mkGaussian(SIZE, COVARIANCE, MEAN, AMPLITUDE)

            #   Compute a matrix with dimensions SIZE (a [Y X] 2-vector, or a
            #   scalar) containing a Gaussian function, centered at pixel position
            #   specified by MEAN (default = (size+1)/2), with given COVARIANCE (can
            #   be a scalar, 2-vector, or 2x2 matrix.  Default = (min(size)/6)^2),
            #   and AMPLITUDE.  AMPLITUDE='norm' (default) will produce a
            #   probability-normalized function.  All but the first argument are
            #   optional.
            #   from Eero Simoncelli, 6/96.
            if GeneralConfig.VERBOSE:
                print('Makes a foa dependent image')

            size = np.array([n_row, n_col])
            cov = (np.min(size)/1.5)**2.5
            conjugate_T_foa = final_foa.conj().T


            foa_filter = mkGaussian(size, cov, conjugate_T_foa, 1)

            foveated_I = np.zeros((n_row, n_col, n_frames))
            foveated_I[:,:,0] = np.multiply(I[:,:,0].astype('double'), foa_filter)
            foveated_I[:,:,1] = np.multiply(I[:,:,1].astype('double'), foa_filter)
            foveated_I[:,:,2] = np.multiply(I[:,:,2].astype('double'), foa_filter)

            # For visualization
            show_foveated_frame = foveated_I[:,:,1]

            # A.2 COMPUTE FEATURES OF THE PHYSICAL LANDSCAPE

            # reducing the frame to [64 64] dimension suitable for feature
            # extraction
            if GeneralConfig.VERBOSE:
                print('Get features')


            Seq=np.zeros((64, 64, n_frames))
            S = resize(foveated_I[:,:,0].astype('double'), (64, 64), order=1) # Bilinear by default
            Seq[:,:,0] = np.divide(S, np.std(S[:]))
            S = resize(foveated_I[:,:,1].astype('double'), (64, 64), order=1)
            Seq[:,:,1] = S / np.std(S[:])
            S = resize(foveated_I[:,:,2].astype('double'), (64, 64), order=1)
            Seq[:,:,2] = S / np.std(S[:])
            Seq = Seq / np.std(Seq[:])


            # The method for computing features has been set in the config_file
            fMap = esComputeFeatures(Seq, feature_type, feature_params)
            # Using the current frame for visualization
            show_feature_map = resize(fMap[:,:,1,1].astype('double'), (n_row, n_col), order=1)
            foveated_feature_map = fMap

            # A.3 SAMPLE THE FOVEATED SALIENCE MAP
            #
            # The method for sampling the salience has been set in the
            # config_file
            if GeneralConfig.VERBOSE:
                print('Sample a saliency map')

            saliency_map = esComputeSalience(foveated_feature_map, Seq, sal_type, salience_params)
            saliency_map = resize(saliency_map[:,], (I.shape[0], I.shape[1]), order=1)

            # For visualization
            # print(curr_frame[:,:,0].shape)
            # print(saliency_map.shape)
            # show_saliency_map = np.dstack((curr_frame, saliency_map))


            # print(show_saliency_map.shape)


        else:
            print('Unknown experiment type')




        numproto = 0; # This will  change only if PROTO=1 and proto-objects sampled
        if ProtoConfig.PROTO:
            # A.4 SAMPLE PROTO-OBJECTS
            # Using the proto-object representation which is the base of method
            # described in IEEE Trans SMC paper [2]
            #
            # If no proto-object are detected or PROTO is false, then we simply
            # go back to the original procedure described in the ICIAP 2011
            # paper [2]

            # Sampling the patch or proto-object map M(t)
            if GeneralConfig.VERBOSE:
                print('Sampling the proto-object map');

            mt_map, protomap, protomap_raw, saliency_norm = esSampleProtoMap(saliency_map)

            # We now have:
            #   the proto-object map                            M(t)
            #   the overlay rapresentation of proto-objects:    protoMap
            #   the raw proto-object map:                       protoMap_raw
            #   the normalized saliency:                        saliency_norm

            show_proto = protomap_raw

            # Sampling the proto-object parameters
            if GeneralConfig.VERBOSE:
                print('Sampling the proto-object parameters')

            num_proto, new_proto_params = esSampleProtoParameters(mt_map, old_protoParam)

            # Saving current parameters
            old_protoParam["B"] = new_proto_params["B"] # The proto-objects boundaries: B{k}
            # the proto-objects fitting ellipses parameters:
            #    a(1)x^2 + a(2)xy + a(3)y^2 + a(4)x + a(5)y + a(6) = 0
            old_protoParam["a"] = new_proto_params["a"]     # conics parameters: a{k}
            # normal form parameters: ((x-cx)/r1)^2 + ((y-cy)/r2)^2 = 1
            old_protoParam["r1"] = new_proto_params["r1"] # axis
            old_protoParam["r2"] = new_proto_params["r2"] # axis
            old_protoParam["cx"] = new_proto_params["cx"] # patch centers
            old_protoParam["cy"] = new_proto_params["cy"] # --
            # Rotated by theta
            old_protoParam["theta"] = new_proto_params["theta"] # Normal form parameters

            # Determine the center and the area of patches for subsequent IP
            # sampling
            cx = new_proto_params["cx"]
            cy = new_proto_params["cy"]
            if num_proto > 0:
                proto_object_centers = np.array([list(cx.values()), list(cy.values())]).T
                nV = proto_object_centers.shape[0]
                if GeneralConfig.VERBOSE:
                    print(f"Number of protObject_centers: {proto_object_centers.shape[0]}")

                show_proto = np.ma.masked_where(protomap == 0, protomap)
                area_proto = np.zeros(nV)


                for p in range(0, nV):
                    # for all proto-objects: area of the fitting ellipse/area of the saliency map
                    area_proto[p] = new_proto_params["r1"][p]*new_proto_params["r2"][p]*np.pi

            # A.5 SAMPLE INTEREST POINTS / PREYS
            # sampling from proto-objects or directly from the map if numproto==0
            if num_proto > 0:
                # Random sampling from proto-objects
                if GeneralConfig.VERBOSE:
                    print("Sample interest points from proto-objects")


                total_area= np.sum(area_proto)

                all_points = np.empty((0, 2), float)
                N=0
                for p in range(nV):
                    # Finds the number of IPs per patch
                    n = round(3 * IPConfig.MAX_POINTS * area_proto[p] / total_area)
                    if n > 0:
                        N += n
                        cov_proto = np.array([[(5*new_proto_params["r2"][p]) , 0],
                                            [0, (5*new_proto_params["r1"][p])]])
                        mu_proto = proto_object_centers[p]
                        # PYMC
                        mv_normal_dist = pm.MvNormal.dist(mu=mu_proto, cov=cov_proto, shape=(2, ))
                        # print(.shape)

                        r_p = mv_normal_dist.random(size=n)
                        all_points = np.vstack((all_points, r_p))


                xCord = all_points[:,0]
                yCord = all_points[:,1]

                if IPConfig.WITH_PERTURBATION:
                    # Adds some other IPs by sampling directly from the salience
                    # landscape
                    xCord2, yCord2, scale = interest_point_sampling(saliency_map)

                    N2 = len(scale) # Number of points
                    # SampledPointsCoord2 = [xCord2, yCord2]

                    N += N2
                    xCord = np.append(xCord, xCord2)
                    yCord = np.append(yCord, yCord2)
            else:
                # If no proto-object have been found sampling from the salience map
                # random sampling weighted by saliency as in  Boccignone & Ferraro [2]
                if GeneralConfig.VERBOSE:
                    print("No patches detected: Sampling interest points")

                xCord, yCord, scale = interest_point_sampling(saliency_map)
                N = len(scale) # Number of points


            sampled_points_coord = [xCord, yCord]

        # B. SAMPLING THE APPROPRIATE OCULOMOTOR ACTION

        # B.1 EVALUATING THE COMPLEXITY OF THE SCENE
        # $$C(t)$$ captures the time-varying configurational complexity of interest points
        # within the landscape

        # Step 1. Computing the 2D histogram of IPs
        # Inputs:
        #     SampledPointsCoord: N x 2 real array containing N data points or N x 1 array
        #     nbinsx:             number of bins in the x dimension (defaults to 20)
        #     nbinsy:             number of bins in the y dimension (defaults to 20)

        if GeneralConfig.VERBOSE:
            print("Histogramming interest points")

        n_bins_x = np.floor(n_row / IPConfig.X_BIN_SIZE).astype(int)
        n_bins_y = np.floor(n_col / IPConfig.Y_BIN_SIZE).astype(int)

        hist_mat, x_bins, y_bins = np.histogram2d(sampled_points_coord[0], sampled_points_coord[1],
                                                  bins=[n_bins_x, n_bins_y])
        # plt.close(hist2d_img)
        #  We now have:
        #      histmat:   2D histogram array (rows represent X, columns represent Y)
        #      Xbins:     the X bin edges (see below)
        #      Ybins:     the Y bin edges (see below)
        num_samples = np.sum(np.sum(hist_mat))
        num_bins = hist_mat.shape[0]*hist_mat.shape[1]

        # Step 2 Evaluate complexity $$C(t)$$

        if GeneralConfig.VERBOSE:
            print('Evaluate complexity')


        disorder, order, complexity = compute_complexity(hist_mat, num_samples, num_bins)

        # Used for plotting the complexity curve in time
        disorder_plot.append(disorder)
        order_plot.append(order)
        complexity_plot.append(complexity)

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

            ax[0, 0].clear()
            ax[0, 1].clear()
            ax[0, 2].clear()
            ax[0, 3].clear()
            ax[0, 4].clear()
            ax[1, 0].clear()
            ax[1, 1].clear()
            ax[1, 2].clear()
            ax[1, 3].clear()
            ax[1, 4].clear()


            # Displaying relevant steps of the process.

            # 1. The original frame.
            ax[0, 0].imshow(show_curr_frame)
            ax[0, 0].set_title('Current frame')

            # subplot(NUMLINE, NUMPICSLINE, countpics);
            # sc(currFrame);
            # label(currFrame, 'Current frame');

            # 2. The foveated frame.
            ax[0, 1].imshow(show_foveated_frame, cmap='gray')
            ax[0, 1].set_title('Foveated frame')

            if GeneralConfig.SAVE_FOV_IMG:
                pass
                # [X,MAP]= frame2im(getframe);
                # FILENAME=[RESULT_DIR VIDEO_NAME '/FOV/FOV' imglist(iFrame).name];
                # imwrite(X,FILENAME,'jpeg');

            # 3. The feature map
            ax[0, 2].imshow(show_feature_map, cmap='gray')
            ax[0, 2].set_title('Feature Map')

            # 4. The saliency map
            ax[0, 3].imshow(saliency_map, cmap='jet')
            ax[0, 3].set_title('Saliency Map')
            if GeneralConfig.SAVE_SAL_IMG:
                pass
                # [X,MAP]= frame2im(getframe);
                # FILENAME=[RESULT_DIR VIDEO_NAME '/SAL/SAL' imglist(iFrame).name];
                # imwrite(X,FILENAME,'jpeg');


            # 5. The proto-objects
            if num_proto > 0:
                ax[0, 4].imshow(show_curr_frame)
                ax[0, 4].imshow(show_proto, cmap='gray', interpolation='nearest')
                ax[0, 4].set_title('Proto-Objects')
                for p in range(nV):
                    ((centx,centy), (width,height), angle) = new_proto_params["a"][p]
                    elli = Ellipse((centx,centy), width, height, angle)
                    elli.set_ec('yellow')
                    elli.set_fill(False)
                    ax[0, 4].add_artist(elli)
                    if GeneralConfig.SAVE_PROTO_IMG:
                        pass
                #         [X,MAP]= frame2im(getframe);
                #         FILENAME=[RESULT_DIR VIDEO_NAME '/PROTO/PROTO' imglist(iFrame).name];
                #         imwrite(X,FILENAME,'jpeg');

            #  6. The Interest points
            ax[1, 0].imshow(show_curr_frame)
            ax[1, 0].set_title("Sampled Interest Points (IP)")
            # Show image with region marked
            for b in range(yCord.shape[0]):
                # plot(yCord(b),xCord(b),'r.')
                circle = Circle((xCord[b],yCord[b]), 4, color='r', lw=1)
                ax[1, 0].add_artist(circle)
                # drawcircle(xCord(b),yCord(b),4,'r',1)

            # for idc in range(len(candx)):
            #     circle = Circle((candx[idc], candy[idc]), 4, color='y', lw=2)
            #     ax[1, 0].add_artist(circle)
            #     # drawcircle(candx(idc), candy(idc),4,'y',2);

            # circle = Circle((candidate_FOA[0], candidate_FOA[1]), 10, color='g', lw=6)
            # ax[1, 0].add_artist(circle)


            if GeneralConfig.SAVE_IP_IMG:
                pass
                # [X,MAP]= frame2im(getframe);
                # FILENAME=[RESULT_DIR VIDEO_NAME '/IP/IP' imglist(iFrame).name];
                # imwrite(X,FILENAME,'jpeg');


            # 7. The IP Empirical distribution for computing complexity
            import seaborn as sns
            sns.heatmap(hist_mat.T, linewidth=0.2, cbar=False, cmap='jet', ax=ax[1, 1])
            ax[1, 1].set_axis_off()
            ax[1, 1].set_title("IP Empirical Distribution")
            if GeneralConfig.SAVE_HISTO_IMG:
                pass
                # [X,MAP]= frame2im(getframe);
                # FILENAME=[RESULT_DIR VIDEO_NAME '/HISTO/HISTO' imglist(iFrame).name];
                # imwrite(X,FILENAME,'jpeg');


            # 8. The Complexity curves
            ax[1, 2].plot(disorder_plot, 'r--', label='Disorder', linewidth=line_width)
            ax[1, 2].plot(order_plot, 'g-', label='Order', linewidth=line_width)
            ax[1, 2].set_title("Order/Disorder")

            ax[1, 3].plot(complexity_plot, label='Complexity', linewidth=line_width)
            ax[1, 3].set_title("Complexity")
            print(disorder_plot)
            print(order_plot)


            plt.tight_layout()
            plt.pause(0.001)
        # break

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

