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
from config import ProtoConfig
import cv2

def esSampleProtoMap(s_map):
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
    M_tMap = esSampleNbestProto(protomap_raw, opening_window)

    proto_map = np.logical_not(M_tMap)

    return M_tMap, proto_map, protomap_raw, norm_sal


def esSampleNbestProto(protoMap_raw, win):
    """Samples the N_V best patches.

    Samples the N_V best patches ranked through their size and returns the actual
    M(t) map

    Args:
        protoMap_raw (matrix): the raw patch map
        win (integer): the window size

    Returns:
       M_tMap (matrix): the patch map M(t)
    """
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (win,win))
    closing = cv2.morphologyEx(protoMap_raw, cv2.MORPH_OPEN, se)

    contours, hierarchy = cv2.findContours(cv2.convertScaleAbs(protoMap_raw),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    size_c = []
    for i,c in enumerate(contours):
        size_c.append(c.shape[0])
    sort_idx = np.argsort(np.array(size_c))[::-1]

    M_tMap = np.zeros(protoMap_raw.shape)
    nBest = min(ProtoConfig.N_BEST_PROTO, len(contours))

    for i in range(nBest):
        img = np.zeros(protoMap_raw.shape)
        cv2.fillPoly(img, pts =[contours[sort_idx[i]]], color=(255,255,255))
        img = img/np.max(img)
        M_tMap = M_tMap + img

    return M_tMap
