import numpy as np

from config import GazeConfig
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


def esGetGazeAttractors(landscape, pred_foa, num_proto):
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
    if num_proto > 0:
        area_proto = landscape["area_proto"]
        prot_object_centers = landscape["proto_centers"]
    else:
        histmat = landscape["histmat"]
        xbin_size = landscape["xbinsize"]
        ybin_size = landscape["ybinsize"]
        NMAX = landscape["NMAX"]

    # Landscape Evaluation
    if GazeConfig.SIMPLE_ATTRACTOR:
        if num_proto > 0:
            # Patch of maximum area to set at least 1 potential candidateFOA mean
            index = np.argmax(area_proto)
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
                X = [x_max, y_max]
                XI = np.flip(pred_foa, axis=1)
                k, d = dsearchn(X,XI)

                # This actually is the column index in the image bitmap
                foa_x = round(x_max[k, 0]*xbin_size - xbin_size/2)
                # This actually is the row index in the image bitmap!
                foa_y = round(y_max[k, 0]*ybin_size - ybin_size/2)

                # Swap image coordinates
                foa_x, foa_y = foa_y, foa_x

        # Now we have at least 1 potential candidate FOA
        # simple one point attractor: use the candidate FOA
        foa_attractors = [foa_x, foa_y]
    else:
        # Multipoint attractor
        if num_proto > 0:
            foa_attractors = np.zeros(prot_object_centers.shape)
            foa_attractors[:,0] = round(prot_object_centers[:,0])
            foa_attractors[:,1] = round(prot_object_centers[:,1])
        else:
        # %find first  NMAX to determine the total attractor potential in
#         %LANGEVIN:
#         %   HI...HNMAX. H=(X-X0)^2 , dH/dX=2(X-X0)

            # the engine
            [ms,mx] = sort(histmat(:),'descend');
            # histmat.flatten()[::-1].sort()
            [rx,cx] = ind2sub(size(histmat),mx)

            # row col val: row col inverted with respect to image coord
            foa_attractors_all = [rx,cx,ms]
            foa_attractors_all = foa_attractors_all[0:NMAX,:]

            # Retains only row col
            foa_attractors = foa_attractors_all[:,0:1]
            # This actually is the column index in the image bitmap
            foa_attractors[:,0] = round(foa_attractors[:,0]*xbin_size - xbin_size/2)
            # This actually is the column index in the image bitmap!
            foa_attractors[:,1] = round(foa_attractors[:,1]*ybin_size - ybin_size/2)

            # Swap image coordinates
            foa_attractors = np.fliplr(foa_attractors, axis=1)
