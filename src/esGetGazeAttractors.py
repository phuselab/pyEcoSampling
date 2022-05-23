import numpy as np

# function FOA_attractors = esGetGazeAttractors(landscape, predFOA, numproto, SIMPLE_ATTRACTOR)

# %   (struct) landscape
# %    - areaProto;
# %    - protObject_centers;
# %        or
# %    - histmat;
# %    - xbinsize;
# %    - ybinsize;
# %    - NMAX;

# % See also
# %   esGazeSampling
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
# % Author
# %   Giuseppe Boccignone <Giuseppe.Boccignone(at)unimi.it>
# %
# % Changes
# %   12/12/2012  First Edition
# %



def esGetGazeAttractors(landscape, predFOA, numproto, SIMPLE_ATTRACTOR):
    """Samples the possible gaze attractors.

    Function computing possible ONE or MULTIPLE gaze attractors
    If a landscape of proto-objects is given then their centers are used as described in [1].
    Otherwise attractors are determined through the IPs sampled from saliency as in [2].

    Args:
        landscape (dict): The parameters for setting the landscape representation
            (proto-objects or straight IPs)
        predFOA (vector): 1 x 2 vector representing the previous FoA coordinates
        numproto (integer): actual number of proto-objects
        SIMPLE_ATTRACTOR (bool): if true (1) using a single best attractor;
            otherwise, determines multiple attractors

    Returns:
        FOA_attractors (matrix):  N_V x 2 matrix representing the FoA attractors
    """

    # If true: if multiple maxima, choose the first closest one
    # to the previous FOA for stability purposes
    MAKE_STABLE = False

    # Setting the landscape
    if numproto:
        area_proto = landscape["areaProto"]
        prot_object_centers = landscape["protObject_centers"]
    else:
        histmat = landscape["histmat"]
        xbin_size = landscape["xbinsize"]
        ybin_size = landscape["ybinsize"]
        NMAX = landscape["NMAX"]


    # Landscape Evaluation
    if SIMPLE_ATTRACTOR:
        if numproto:
            # Patch of maximum area to set at least 1 potential candidateFOA mean
            index = np.argmax(area_proto)
            max_proto = area_proto[index]
            # Center fo the Patch
            foa_x = round(prot_object_centers[index, 0])
            foa_y = round(prot_object_centers[index, 1])
        else:
            # Histogram maximum to set at least 1 potential candidateFOA mean
            max_hist = np.max(np.max(histmat))
            x_max, y_max = np.where(histmat == max_hist)
            k = 1
            if MAKE_STABLE:
                # If multiple maxima, choose the first closest one
                # to the previous FOA for stability purposes
                X = (x_max, y_max)
                XI =

#     else
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
