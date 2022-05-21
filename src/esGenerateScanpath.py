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
# error(nargchk(1, 2, nargin));
# if ~exist('nOBS', 'var') || isempty(nOBS)
#     nOBS = 1;
# end

# try
#     eval(config_file);
# catch ME
#     disp('error in evaluating config script')
#     ME
# end

# %% INITIALIZATION
# % The EXPERIMENT_TYPE sets Feature Extraction and Salience Map methods
# %
# % Setting feature parameters
# if strcmp(EXPERIMENT_TYPE,'3DLARK_SELFRESEMBLANCE')
#     % Set 3-D LARKs Parameters
#     fType          = EXPERIMENT_TYPE;
#     fParam.wsize   = WSIZE; % LARK spatial window size
#     fParam.wsize_t = WSIZE_T; % LARK temporal window size
#     fParam.alpha   = LARK_ALPHA; % LARK sensitivity parameter
#     fParam.h       = LARK_H;  % smoothing parameter for LARK
#     fParam.sigma   = LARK_SIGMA; % fall-off parameter for self-resemblamnce
# else
#         error('Unknown FEATURE type');
# end

# % Setting salience parameters
# if strcmp(EXPERIMENT_TYPE,'3DLARK_SELFRESEMBLANCE')
#     % Set Self Resemblance Parameters
#     salType        = EXPERIMENT_TYPE;
#     sParam.wsize   = WSIZE; % LARK spatial window size
#     sParam.wsize_t = WSIZE_T; % LARK temporal window size
#     sParam.sigma   = LARK_SIGMA; % fall-off parameter for self-resemblamnce
# else
#         error('Unknown SALIENCE type');
# end

# % Set proto object structure
# old_protoParam.B     = [];
# old_protoParam.a     = [];
# old_protoParam.r1    = [];
# old_protoParam.r2    = [];
# old_protoParam.cx    = [];
# old_protoParam.cy    = [];
# old_protoParam.theta = [];

# % Setting parameters for the $$Dirichlet(\pi; nu0,nu1,nu2)$$ distribution
# nu(1)=1; nu(2)=1; nu(3)=1; % we start with equal probabilities

# % Setting sampling parameters

# % Internal simulation: somehow related to visibility: the more the points
# % that can be sampled the higher the visibility of the field
# gazeSampParam.NUM_INTERNALSIM = NUM_INTERNALSIM; %maximum allowed number of candidate new  gaze position r_new
# % If anything goes wrong retry:
# gazeSampParam.MAX_NUMATTEMPTS = MAX_NUMATTEMPTS; %maximum allowed tries for sampling e new valid gaze position



# % Setting parameters for the alpha-stable distribution
# alpha_stblParam.alpha = alpha_stable;
# alpha_stblParam.beta  = beta_stable;
# alpha_stblParam.gamma = gamma_stable;
# alpha_stblParam.delta = delta_stable;

# % Allocating some vectors to be used later
# Disorderplot=[]; Orderplot=[]; Complplot=[];
# if SAVE_FOA_ONFILE
#     allFOA=[];
# end

# % Reads the first frame and gets some parameters: frame size, etc...
# imglist = dir([FRAME_DIR '*.jpg']);
# fnum    = length(imglist);
# im1     = imread([FRAME_DIR imglist(nNImageStart-1).name]);
# [r c]   = size(im1(:,:,1)); % rows, columns
# nrow=r;
# ncol=c;

# % Set 2D histogram structure
# nbinsx=fix(nrow/xbinsize); nbinsy=fix(ncol/ybinsize);
# XLO = 1; XHI = nrow; YLO = 1; YHI =ncol;

# % Set first FOA: default frame center
# foaSize = round(max(r,c) / 6);
# if firstFOAonCENTER
#         xc=round(r/2);
#         yc=round(c/2);
# else
#     fprintf('\n No methods defined for setting the  first FOA');
# end
# finalFOA = [xc yc];

# % Set visualization parameters
# if VISUALIZE_RESULTS
#     font_size=18; line_width=2 ; marker_size=16;
#     NUMLINE=2;
#     NUMPICSLINE=5;
#     % Set up display window
#     scrsz = get(0,'ScreenSize');
#     figure('Position',[1 scrsz(4) scrsz(3) scrsz(4)],'Name','Ecological Sampling of Gaze Shift Demo','NumberTitle','off')
# end


# n        = 0; % number of iterations
# numproto = 0; % number of proto_objects
# dir_old  = 0; % previous gaze shift direction

# %%THE ECOLOGICAL SAMPLING CYCLE UNFOLDING IN TIME
# %
# for iFrame=nNImageStart+offset:frameStep:nNImageEnd

#     fprintf('\n Processing Frame #%d\n', iFrame);

#     % Choosing features and salience to be used within the experiment
#     if strcmp(EXPERIMENT_TYPE,'3DLARK_SELFRESEMBLANCE')
#         nFrames=WSIZE_T;
#         n=n+1;

#         %%A. SAMPLING THE NATURAL HABITAT (VIDEO)
#         %
#         % Reading three consecutive frames
#         if VERBOSE
#             disp('Data acquisition')
#         end
#         predFrame = imread([FRAME_DIR imglist(iFrame-offset).name]);  % get previous frame
#         currFrame = imread([FRAME_DIR imglist(iFrame).name]);         % get current frame
#         nextFrame = imread([FRAME_DIR imglist(iFrame+offset).name]);  % get subsequent frame

#         % Converting to grey level
#         I        = zeros(r,c,nFrames);
#         I(:,:,1) = rgb2gray(predFrame);
#         I(:,:,2) = rgb2gray(currFrame);
#         I(:,:,3) = rgb2gray(nextFrame);

#         % For visualization
#         show_currFrame=I(:,:,2);

#         %%A.1 MAKES A FOVEATED FRAME
#         %
#         % Using:
#         %   IM = mkGaussian(SIZE, COVARIANCE, MEAN, AMPLITUDE)
#         %
#         %   Compute a matrix with dimensions SIZE (a [Y X] 2-vector, or a
#         %   scalar) containing a Gaussian function, centered at pixel position
#         %   specified by MEAN (default = (size+1)/2), with given COVARIANCE (can
#         %   be a scalar, 2-vector, or 2x2 matrix.  Default = (min(size)/6)^2),
#         %   and AMPLITUDE.  AMPLITUDE='norm' (default) will produce a
#         %   probability-normalized function.  All but the first argument are
#         %   optional.
#         %   from Eero Simoncelli, 6/96.
#         if VERBOSE
#             disp('Makes a foa dependent image')
#         end
#         sz=[r,c];   cov=(min(sz(1),sz(2))/1.5)^2.5;
#         foaFilter = mkGaussian(sz, cov, finalFOA',1);
#         foveated_I(:,:,1)= double(I(:,:,1)).*foaFilter;
#         foveated_I(:,:,2)= double(I(:,:,2)).*foaFilter;
#         foveated_I(:,:,3)= double(I(:,:,3)).*foaFilter;

#         %For visualization
#         show_foveatedFrame = foveated_I(:,:,2);

#         %%A.2 COMPUTE FEATURES OF THE PHYSICAL LANDSCAPE
#         %
#         % reducing the frame to [64 64] dimension suitable for feature
#         % extraction
#         if VERBOSE
#             disp('Get features')
#         end
#         S = imresize(double(foveated_I(:,:,1)),[64 64],'bilinear');
#         Seq(:,:,1) = S/std(S(:));
#         S = imresize(double(foveated_I(:,:,2)),[64 64],'bilinear');
#         Seq(:,:,2) = S/std(S(:));
#         S = imresize(double(foveated_I(:,:,3)),[64 64],'bilinear');
#         Seq(:,:,3) = S/std(S(:));
#         Seq = Seq/std(Seq(:));

#         % The method for computing features has been set in the
#         % config_file
#         fMap = esComputeFeatures(Seq, fType, fParam);

#         % using the current frame for visualization
#         show_fMap = imresize(double(fMap(:,:,2)),[nrow ncol],'bilinear');
#         foveated_fMap = fMap;

#         %%A.3 SAMPLE THE FOVEATED SALIENCE MAP
#         %
#         % The method for sampling the salience has been set in the
#         % config_file
#         if VERBOSE
#             disp('Sample a saliency map')
#         end
#         sMap  = esComputeSalience(foveated_fMap, Seq, salType, sParam);
#         sMap  = imresize(sMap(:,:),[size(I,1) size(I,2)]);

#         % for visualization
#         show_sMap = sMap;
#     else
#         error('Unknown experiment type');
#     end

#     numproto = 0; %this will  change only if PROTO=1 and proto-objects sampled
#     if PROTO
#         %%A.4 SAMPLE PROTO-OBJECTS
#         % Using the proto-object representation which is the base of method
#         % described in IEEE Trans SMC paper [2]
#         %
#         % If no proto-object are detected or PROTO is false, then we simply
#         % go back to the original procedure described in the ICIAP 2011
#         % paper [2]
#         %

#         % Sampling the patch or proto-object map M(t)
#         if VERBOSE
#             fprintf('\n Sampling the proto-object map \n');
#         end
#         [M_tMap protoMap protoMap_raw sal] = esSampleProtoMap(sMap,currFrame,nBestProto);
#         % we now have:
#         %   the proto-object map                            M(t)
#         %   the overlay rapresentation of proto-objects:    protoMap
#         %   the raw proto-object map:                       protoMap_raw
#         %   the normalized saliency:                        sal

#         show_proto = protoMap_raw;

#         % Sampling the proto-object parameters
#         if VERBOSE
#             fprintf('\n Sampling the proto-object parameters \n');
#         end
#         [numproto new_protoParam] = esSampleProtoParameters(M_tMap, old_protoParam);

#         % Saving current parameters
#         old_protoParam.B     = new_protoParam.B;     % the proto-objects boundaries: B{k}
#         % the proto-objects fitting ellipses parameters:
#         %    a(1)x^2 + a(2)xy + a(3)y^2 + a(4)x + a(5)y + a(6) = 0
#         old_protoParam.a     = new_protoParam.a;     % conics parameters: a{k}
#         % normal form parameters: ((x-cx)/r1)^2 + ((y-cy)/r2)^2 = 1
#         old_protoParam.r1    = new_protoParam.r1;    % axis
#         old_protoParam.r2    = new_protoParam.r2;    % axis
#         old_protoParam.cx    = new_protoParam.cx;    % patch centers
#         old_protoParam.cy    = new_protoParam.cy;    % --
#         % rotated by theta
#         old_protoParam.theta = new_protoParam.theta; % normal form parameters

#         % Determine the center and the area of patches for subsequent IP
#         % sampling
#         % fprintf('\n Proto-objects centers x=%f y=%f \n', cx, cy);
#         if numproto > 0
#             protObject_centers = [new_protoParam.cx(:), new_protoParam.cy(:)];
#             nV = size(protObject_centers,1);
#             if VERBOSE
#                 fprintf('\n Number of protObject_centers: %d \n', size(protObject_centers,1));
#             end
#             show_proto = imoverlay(currFrame, protoMap, [0 0 0]);

#             areaProto=zeros(1,nV);
#             %totArea=size(sal,1)*size(sal,2);
#             %kmax=nV;
#             for p=1:nV
#                  %for all proto-objects: area of the fitting ellipse/area of the saliency map
#                  areaProto(p)= new_protoParam.r1(p)*new_protoParam.r2(p)*pi;
#             end
#         end
#     end

#     %%A.5 SAMPLE INTEREST POINTS / PREYS
#     % sampling from proto-objects or directly from the map if numproto==0
#     if numproto >0
#         % Random sampling from proto-objects
#         if VERBOSE
#             fprintf('\n Sample interest points from proto-objects \n');
#         end

#         %totArea=size(sal,1)*size(sal,2);
#         totArea=sum(areaProto);

#         allpoints=[];
#         N=0;
#         for p=1:nV
#             %finds the number of IPs per patch
#             n = round(3*Interest_Point.Max_Points * areaProto(p)/totArea);
#             N = N+n;

#             % Samples interest points from a Normal centered on the patch
#             % Using:
#             %  RANDNORM(n,m,[],V) returns a matrix of n columns where each column is a sample
#             %                     from a multivariate normal with mean m (a column vector) and
#             %                     V specifies the covariance matrix.
#             covProto  = [(5*new_protoParam.r2(p)) , 0; 0, (5*new_protoParam.r1(p))];
#             muProto   = [protObject_centers(p,1); protObject_centers(p,2)];
#             r_p       = randnorm(n, muProto, [], covProto);
#             allpoints = [allpoints r_p];

#         end
#         SampledPointsCoord= allpoints';
#         xCord= SampledPointsCoord(:,1);
#         yCord= SampledPointsCoord(:,2);

#         if WITH_PERTURBATION
#             % Adds some other IPs by sampling directly from the salience
#             % landscape
#             [xCord2 yCord2 scale]= InterestPoint_Sampling(sMap,Interest_Point);
#             N2=length(scale); %number of points
#             SampledPointsCoord2=[xCord2 yCord2];

#             N=N+N2;

#             xCord=[xCord;xCord2];
#             yCord=[yCord;yCord2];
#         end
#     else
#         % If no proto-object have been found sampling from the salience map
#         % random sampling weighted by saliency as in  Boccignone & Ferraro [2]
#         if VERBOSE
#             fprintf('\n No patches detected: Sampling interest points \n');
#         end
#         [xCord yCord scale]= InterestPoint_Sampling(sMap,Interest_Point);
#         N=length(scale); %number of points

#     end

#     SampledPointsCoord=[xCord yCord];

#     %%B. SAMPLING THE APPROPRIATE OCULOMOTOR ACTION

#     %%B.1 EVALUATING THE COMPLEXITY OF THE SCENE
#     % $$ C(t)$$ captures the time-varying configurational complexity of interest points
#     % within the landscape

#     % Step 1. Computing the 2D histogram of IPs
#     % Inputs:
#     %     SampledPointsCoord: N x 2 real array containing N data points or N x 1 array
#     %     nbinsx:             number of bins in the x dimension (defaults to 20)
#     %     nbinsy:             number of bins in the y dimension (defaults to 20)
#     if VERBOSE
#         fprintf('\n Histogramming interest points \n');
#     end

#     [histmat Xbins Ybins] = hist2d(SampledPointsCoord,nbinsx, nbinsy,  [XLO XHI], [YLO YHI]);
#     % we now have:
#     %     histmat:   2D histogram array (rows represent X, columns represent Y)
#     %     Xbins:     the X bin edges (see below)
#     %     Ybins:     the Y bin edges (see below)

#     numSamples = sum(sum(histmat));
#     numBins    = size(histmat,1)*size(histmat,2);

#     % Step2. Evaluate complexity $$ C(t)$$
#     %
#     if VERBOSE
#         fprintf('\n Evaluate complexity \n');
#     end
#     [Disorder Order Compl] = esComputeComplexity(complexity_type, histmat, numSamples, numBins);

#     % Used for plotting the complexity curve in time
#     Disorderplot=[Disorderplot Disorder]; Orderplot=[Orderplot Order]; Complplot=[Complplot Compl];

#     %%B.2. ACTION SELECTION VIA LANDSCAPE COMPLEXITY

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

#     %%C. SAMPLING THE GAZE SHIFT
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

#     if VISUALIZE_RESULTS
#         %%THE ART CORNER
#         countpics=1;
#         % Displaying relevant steps of the process.
#         %figure(1);

#         % 1. The original frame.
#         subplot(NUMLINE, NUMPICSLINE, countpics); sc(currFrame); label(currFrame, 'Current frame');

#         % 2. The foveated frame.
#         countpics=countpics+1;
#         subplot(NUMLINE, NUMPICSLINE, countpics);  sc(show_foveatedFrame); label(show_foveatedFrame, 'Foveated Frame')
#         if SAVE_FOV_IMG
#             [X,MAP]= frame2im(getframe);
#             FILENAME=[RESULT_DIR VIDEO_NAME '/FOV/FOV' imglist(iFrame).name];
#             imwrite(X,FILENAME,'jpeg');
#         end

#         % 3. The feature map
#         countpics=countpics+1;
#         subplot(NUMLINE, NUMPICSLINE, countpics); sc(show_fMap); label(show_fMap, 'Feature Map')

#         % 4. The saliency map
#         countpics=countpics+1;
#         subplot(NUMLINE, NUMPICSLINE, countpics) ;
#         tempIm=cat(3,show_sMap, show_currFrame); sc(tempIm,'prob_jet');  label(tempIm, 'Saliency map');
#         if SAVE_SAL_IMG
#             [X,MAP]= frame2im(getframe);
#             FILENAME=[RESULT_DIR VIDEO_NAME '/SAL/SAL' imglist(iFrame).name];
#             imwrite(X,FILENAME,'jpeg');
#         end

#         % 5. The proto-objects
#         if numproto>0
#             countpics=countpics+1;
#             subplot(NUMLINE, NUMPICSLINE, countpics);
#             sc(show_proto); hold on
#             for p=1:nV
#                     [X Y]=drawellip2(new_protoParam.a{p});
#                     hold on
#                     plot(Y,X,'y')
#             end
#             label(show_proto, 'Proto-Objects');
#             if SAVE_PROTO_IMG
#                 [X,MAP]= frame2im(getframe);
#                 FILENAME=[RESULT_DIR VIDEO_NAME '/PROTO/PROTO' imglist(iFrame).name];
#                 imwrite(X,FILENAME,'jpeg');
#             end
#         end
#         hold off

#         % 6. The Interest points
#         countpics=countpics+1;
#         subplot(NUMLINE, NUMPICSLINE, countpics);
#         sc(currFrame); label(currFrame, 'Sampled Interest Points (IP)');
#         %%% Show image with region marked
#         hold on;
#         for b=1:size(yCord,1)
#             plot(yCord(b),xCord(b),'r.');
#             drawcircle(xCord(b),yCord(b),4,'r',1);
#             hold on;
#         end

#         for idc=1:length(candx)
#                 drawcircle(candx(idc), candy(idc),4,'y',2);
#                  hold on;
#         end

#         drawcircle(candidateFOA(1), candidateFOA(2),10,'g',6);hold on;

#         if SAVE_IP_IMG
#              [X,MAP]= frame2im(getframe);
#              FILENAME=[RESULT_DIR VIDEO_NAME '/IP/IP' imglist(iFrame).name];
#              imwrite(X,FILENAME,'jpeg');
#         end
#         hold off;
#         hold off;

#         % 7. The IP Empirical distribution for computing complexity
#         countpics=countpics+1;
#         subplot(NUMLINE, NUMPICSLINE, countpics);
#         pcolor(Ybins,Xbins,flipud(histmat'));
#         axis square tight ;
#         if SAVE_HISTO_IMG
#              [X,MAP]= frame2im(getframe);
#              FILENAME=[RESULT_DIR VIDEO_NAME '/HISTO/HISTO' imglist(iFrame).name];
#              imwrite(X,FILENAME,'jpeg');
#          end


#         % 8. The Complexity curves
#         countpics=countpics+1;
#         hnd_comp=subplot(NUMLINE, NUMPICSLINE, countpics);
#         set(hnd_comp, 'FontSize', font_size); title('Order/Disorder'), hold on ;
#         plot(Disorderplot,'r--','LineWidth',line_width); hold on
#         plot(Orderplot,'g-','LineWidth',line_width); hold on
#         hold off
#         countpics=countpics+1;
#         hnd_comp=subplot(NUMLINE, NUMPICSLINE, countpics);
#         set(hnd_comp, 'FontSize', font_size); title('Complexity'), hold on ;
#         plot(Complplot,'b-'); hold on
#         hold off;

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


