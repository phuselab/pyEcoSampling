# function    [finalFOA dir_new accept candx candy] = esLangevinSimSampling(sdeParam, predFOA, nrow,ncol,...
#                                                                            xCordIP, yCordIP)
# %LangevinSimSampling - Langevin step for sampling the new gaze position
# %
# %
# % Synopsis
# %         [finalFOA dir_new accept candx candy ] = esLangevinSimSampling(sdeParam, predFOA, nrow,ncol,...
# %                                                                        xCordIP, yCordIP)
# %
# % Description
# %     Implements a step of the
# %     Langevin like stochastic differential equation (SDE),
# %     whose noise source is sampled from a mixture of $$\alpha$$-stable distributions.
# %
# %
# % Inputs ([]s are optional)
# %   (struct) sdeParam
# %   -(scalar) dHx                        SDE gradient potential  x coordinate
# %   -(scalar) dHy                        SDE gradient potential  y coordinate
# %   -(scalar) xi                         SDE alpha-stable component
# %   -(scalar) dir_new                    SDE new direction of the gaze shift
# %   -(scalar) alpha                      SDE alpha-stable characteristic exponent parameter
# %   -(scalar) gamma                      SDE alpha-stable scale parameter
# %   -(scalar) h                          SDE integration step
# %   (vector) predFOA                    1 x 2 vector representing the previous FoA coordinates
# %   (scalar) nrow                       number of rows of the map
# %   (scalar) ncol                       number of columns of the map
# %   (vector) xCordIP                    coordinates of IPs
# %   (vector) yCordIP
# %
# %
# % Outputs ([]s are optional)
# %
# %   (vector) finalFOA                   1 x 2 vector representing the new FoA coordinates
# %   (scalar) dir_new                    the new direction of the gaze shift
# %   (bool)   accept = 1                 gaze position is retained as valid
# %   (vector) candx                       coordinates of candidate FOAs from internal simulation
# %   (vector) candy
# %
# %
# % Example:
# %
# %
# %
# % See also
# %   esProtoGazePointSampling
# %
# % Requirements
# %
# %
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
# % Changes
# %   12/12/2012  First Edition
# %

#     VERBOSE = false;
#     candx=[]; candy=[];

#     accept = 0;

#     finalFOA = predFOA; %rough initialization: nothing changes
#     % setting Langevin SDE parameters
#     dHx          = sdeParam.dHx;    %SDE gradient potential  x coordinate
#     dHy          = sdeParam.dHy;    %SDE gradient potential  y coordinate
#     xi           = sdeParam.xi;     %SDE alpha-stable component
#     dir_new      = sdeParam.dir_new;%SDE new direction of the gaze shift
#     alpha_stable = sdeParam.alpha;  %SDE alpha-stable characteristic exponent parameter
#     gamma_stable = sdeParam.gamma;  %SDE alpha-stable scale parameter
#     h            = sdeParam.h;      %SDE integration step


#     % Sampling the shift dx, dy on the basis of the generalized discretized Langevin:
#     % $$\mathbf{r}_{F}(t_{n+1}) \approx \mathbf{r}_{F}(t_{n})  -  \sum_{p=1}^{N_V} ( \mathbf{r}_{F}(t_{n}) -\mathbf{r}_p(t_{n}) ) \tau
#     %    + \gamma_k \mathbb{I} \tau^{1/\alpha_k} \xi_{k}$$

#     value = sqrt(gamma_stable)*(h^(1/alpha_stable))*xi;
#     dx  =   dHx*h   +   value.*sin(dir_new);
#     dy  =   dHy*h   +   value.*cos(dir_new);

#     % Candidate new FOA
#     % (1 x NUM_SIMATTEMPTS) coordinate vectors
#     tryFOA_x = predFOA(1) + dx;
#     tryFOA_y = predFOA(2) + dy;

#     tryFOA_x = round(tryFOA_x);
#     tryFOA_y = round(tryFOA_y);

#     % Verifies if the candidate shift is located within the image
#     validcord = find((tryFOA_x >=1) & (tryFOA_x <nrow) & (tryFOA_y >=1) & (tryFOA_y <ncol));
#     if VERBOSE
#          fprintf('\n Sampled %d valid candidate FOAs', numel(validcord));
#     end
#     if numel(validcord) == 1
#         % Retains only the valid ones
#         tryFOA_x = tryFOA_x(validcord);
#         tryFOA_y = tryFOA_y(validcord);

#         % always make sure these are integer coordinates
#         tryFOA_x = round(tryFOA_x);
#         tryFOA_y = round(tryFOA_y);

#         NcandFOAs =length(tryFOA_y);
#         candx = tryFOA_x; candy = tryFOA_y;
#         finalFOA(1) = candx; finalFOA(2) = candy;
#         finalFOA = round(finalFOA);
#         accept = 1;
#     end

#     if numel(validcord)>1
#         % Retains only the valid ones
#         tryFOA_x = tryFOA_x(validcord);
#         tryFOA_y = tryFOA_y(validcord);

#         % always make sure these are integer coordinates
#         tryFOA_x = round(tryFOA_x);
#         tryFOA_y = round(tryFOA_y);

#         NcandFOAs =length(tryFOA_y);

#         if VERBOSE
#                 fprintf('\n Sampled  %d candidate new FOAS', NcandFOAs);
#         end

#         candx = tryFOA_x; candy = tryFOA_y;

#         % computes the local visibility with respect to the FOA
#         foaSize = round(max(nrow,nrow) / 6);
#         localVisibilityRadius = 1.5*foaSize;

#         % Choose the best FOA among the simulated
#         % ... For each simulated FOA, computes how many preys  /IPs (generate dalle patch)
#         % ....are within the visual search range
#         % ... The IP which can get more preys survives...
#         %
#         % NOTE: this could be improved or extended with smarter criteria!!
#         %
#         sampledIP = [xCordIP yCordIP];
#         candidateFOAs = [tryFOA_x' tryFOA_y'];

#         if VERBOSE
#                 fprintf('\n Choosing the best one');
#         end

#         % Performs a range search via kd-tree:
#         % idxIP contains the indices of points in X
#         % whose distance to candidateFOAs(I,:) are not greater than localVisibilityRadius,
#         % and these indices are sorted in the ascending order of the corresponding distance values D.
#         % - idxIP is NcandFOAs-by-1 cell array, where NcandFOAs is
#         % the number of rows in candidateFOAs
#         % - D{I} contains the distance values between candidateFOAs(I,:) and the corresponding
#         % points returned in idxIP{I}.

#         [idxIP, D] = rangesearch(sampledIP,candidateFOAs,localVisibilityRadius);
#         maxnIP= -1;
#         for nCand=1:NcandFOAs
#             temp=idxIP{nCand};
#             len=length(temp); %get the number of interest points in the visibility range
#             if (len > maxnIP)
#                 maxnIP = len; bestCandID = nCand;
#             end
#         end

#         finalFOA = candidateFOAs(bestCandID, :);
#         finalFOA = round(finalFOA);
#         accept = 1;
#     end
