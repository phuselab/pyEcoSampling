# function FOA_attractors = esGetGazeAttractors(landscape, predFOA, numproto, SIMPLE_ATTRACTOR)

# %esGetGazeAttractors - Samples the possible gaze attractors
# %
# %
# % Synopsis
# %          FOA_attractors = esGetGazeAttractors(landscape, predFOA, numproto, SIMPLE_ATTRACTOR)
# %
# % Description
# %     Function computing possible ONE or MULTIPLE gaze attractors
# %     If a landscape of proto-objects is given then their centers are used as described in [1]
# %     Otherwise attractors are determined through the IPs sampled from saliency as in [2].
# %
# %
# % Inputs ([]s are optional)
# %   (struct) landscape                  The parameters for setting the
# %                                       landscape representation (proto-objects or straight IPs
# %    - areaProto;
# %    - protObject_centers;
# %        or
# %    - histmat;
# %    - xbinsize;
# %    - ybinsize;
# %    - NMAX;
# %   (vector) predFOA                    1 x 2 vector representing the previous FoA coordinates
# %   (scalar) numproto                   actual number of proto-objects
# %   (bool)   SIMPLE_ATTRACTOR           if true (1) using a single best
# %                                       attractor; otherwise, determines multiple attractors
# %
# % Outputs ([]s are optional)
# %
# %   (matrix) FOA_attractors             N_V x 2 matrix representing the
# %                                        FoA attractors
# %
# % Example:
# %
# %
# %
# % See also
# %   esGazeSampling
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

# MAKE_STABLE= false;  % if true: if multiple maxima, choose the first closest one to the previous FOA for stability
#                      % purposes

# % Setting the landscape
# if numproto
#    areaProto          = landscape.areaProto;
#    protObject_centers = landscape.protObject_centers;
# else
#    histmat  = landscape.histmat;
#    xbinsize = landscape.xbinsize;
#    ybinsize = landscape.ybinsize;
#    NMAX     = landscape.NMAX;
# end


# % LANDSCAPE EVALUATION
# if SIMPLE_ATTRACTOR
#     if numproto
#         % patch of maximum area to set at least 1 potential candidateFOA mean
#         maxProto = max(areaProto);
#         ind      = find(maxProto==areaProto);
#         % center of the patch
#         foax     = round(protObject_centers(ind,1));
#         foay     = round(protObject_centers(ind,2));

#     else
#         % histogram maximum to set at least 1 potential candidateFOA mean
#         maxhist     = max(max(histmat));
#         [xmax ymax] = find(maxhist==histmat);
#         k=1;
#         if MAKE_STABLE
#             % if multiple maxima, choose the first closest one to the previous FOA for stability
#             % purposes
#             X           = [xmax ymax];
#             XI          = fliplr(predFOA);%flipping true coordinates because XI are inverted
#             [k,d]       = dsearchn(X,XI);
#         end
#         foax        = round(xmax(k,1)*xbinsize - xbinsize/2);  %this actually is the column index in the image bitmap!
#         foay        = round(ymax(k,1)*ybinsize - ybinsize/2);  %this actually is the row index in the image bitmap!

#         % swap image coordinates
#         temp=foax; foax=foay; foay=temp;
#     end
#     % now we have at least 1 potential candidate FOA
#     % simple one point attractor: use the candidate FOA
#     FOA_attractors = [foax foay];

# else
#     % multipoint attractor
#     if numproto
#         FOA_attractors      = zeros(size(protObject_centers));
#         FOA_attractors(:,1) = round(protObject_centers(:,1));
#         FOA_attractors(:,2) = round(protObject_centers(:,2));
#     else
#         %find first  NMAX to determine the total attractor potential in
#         %LANGEVIN:
#         %   HI...HNMAX. H=(X-X0)^2 , dH/dX=2(X-X0)

#         % the engine
#         [ms,mx] = sort(histmat(:),'descend');
#         [rx,cx] = ind2sub(size(histmat),mx);

#         FOA_attractors_all = [rx,cx,ms]; % row col val: row col inverted with respect to image coord
#         FOA_attractors_all = FOA_attractors_all(1:NMAX,:);

#         % retains only row col
#         FOA_attractors      = FOA_attractors_all(:,1:2);
#         FOA_attractors(:,1) = round(FOA_attractors(:,1)*xbinsize - xbinsize/2);  %this actually is the column index in the image bitmap!
#         FOA_attractors(:,2) = round(FOA_attractors(:,2)*ybinsize - ybinsize/2);  %this actually is the column index in the image bitmap!

#         %swap image coordinates
#         FOA_attractors = fliplr(FOA_attractors);
#     end
# end
