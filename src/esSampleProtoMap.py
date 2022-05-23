# % Requirements
# %   Image Processing toolbox
# %
# % References
# %   [1] G. Boccignone and M. Ferraro, Ecological Sampling of Gaze Shifts
# %       IEEE Trans. Systems Man Cybernetics - Part B (to appear)
# %
# % Author
# %   Giuseppe Boccignone <Giuseppe.Boccignone(at)unimi.it>
# %
# % Changes
# %   12/12/2012  First Edition
# %

import numpy as np

def esSampleProtoMap(s_map, curr_frame, n_best_proto):
    """Generates the patch map M(t).

    In a first step generates the raw patch map by thresholding  the normalized salience map
    so as to achieve 95% significance level for deciding whether the given saliency values are
    in the extreme tails
    In a second step get the N_V best patches ranked through their size and returns the actual
    M(t) map

    Args:
        s_map (matrix): the salience map, 0/1 overlay representation
        curr_frame (matrix): the current frame
        n_best_proto (integer): the N_V most valuable patches

    Outputs:
        M_tMap (matrix): the patch map M(t)
        proto_map (matrix): the object layer representation of patch map M(t)
        proto_map_raw (matrix): the raw patch map
        norm_sal (matrix): the normalized salience map

    Returns:
        _type_: _description_
    """

    protomap_raw = np.zeros(s_map.shape)
    # Normalizing salience
    norm_sal = s_map
    max_sal = np.max(norm_sal)
    min_sal = np.min(norm_sal)
    norm_sal = np.divide((norm_sal-min_sal),(max_sal-min_sal))
    norm_sal = norm_sal*100

    # Method percentile based
    ind = np.stack(np.where(norm_sal >= np.percentile(norm_sal,95)), axis=-1)
    protomap_raw[ind[:,0], ind[:,1]] = 1

    # Samples the N_V best patches
    opening_window=7
    M_tMap = esSampleNbestProto(protomap_raw, n_best_proto, opening_window)

    protoMap = np.logical_not(M_tMap)

    return M_tMap, protoMap

    # ind=find(normSal >= prctile(normSal(:),90));
    # protoMap_raw(ind)=1;
    # protoMap_raw(ind)= currFrame(ind);



    # openingwindow=7;
    # M_tMap = esSampleNbestProto(protoMap_raw,nBestProto,openingwindow);


    # protoMap= M_tMap;
    # fbw = M_tMap==0;
    # protoMap(fbw) = 1;
    # fbw = M_tMap==1;
    # protoMap(fbw) = 0;
    # end



def esSampleNbestProto(protoMap_raw, nBest, win):
    """Samples the N_V best patches.

    Samples the N_V best patches ranked through their size and returns the actual
    M(t) map

    Args:
        protoMap_raw (matrix): the raw patch map
        nBest (integer): the N_V most valuable patches
        win (integer): the window size

    Returns:
       M_tMap (matrix): the patch map M(t)
    """
    pass

# % Use morphological operations
# se = strel('disk',win);
# protoMap_raw = imopen(protoMap_raw,se);

# % Calculating 8 connected components via Rosenfeld and Pfaltz
# [L,obj] = bwlabel(protoMap_raw,8);

# % Returns the foreground connected component in the binary image
# % supplied that have the specified ranked size(s).
# maxL = max(L(:));					      % max number of connected components
# h = hist(L(find(protoMap_raw)),[1:maxL]); % find indexes of labeled elements, by column scanning,
#                                           %    and compute occurence of labels from 1 to maxL
# [sh,sr] = sort(-h);                       % sorting with respect to the number of occurrence
#                                           %    sh is the num occurrences (negative: -8   -6  -4  -2),
#                                           %    sr = corresponding labels
# M_tMap = protoMap_raw & 0;
# if nBest > maxL
#      nBest = maxL;
# end
# for i=1:nBest
#     M_tMap = M_tMap | (L==sr(i)) ;      % returns the nBest by  dimension
# end
# end
