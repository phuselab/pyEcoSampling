"""

Authors:
    - Giuseppe Boccignone <giuseppe.boccignone@unimi.it>
    - Renato Nobre <renato.avellarnobre@studenti.unimi.it>

Changes:
    - 12/12/2012  First Edition Matlab
    - 31/05/2022  Python Edition
"""


from config import ProtoConfig
from utils.logger import Logger
import cv2
from skimage import measure
import numpy as np

logger = Logger(__name__)


class ProtoParameters:

    def __init__(self):
        self.B = []
        self.a = []
        self.r1 = []
        self.r2 = []
        self.cx = []
        self.cy = []
        self.theta = []
        self.show_proto = None
        self.n_best_proto = ProtoConfig.N_BEST_PROTO
        self.area_proto = None
        self.nV = 0
        self.proto_centers = None


            # # the proto-objects fitting ellipses parameters:
            # #    a(1)x^2 + a(2)xy + a(3)y^2 + a(4)x + a(5)y + a(6) = 0
            # old_protoParam["a"] = new_proto_params["a"]     # conics parameters: a{k}
            # # normal form parameters: ((x-cx)/r1)^2 + ((y-cy)/r2)^2 = 1
            # old_protoParam["r1"] = new_proto_params["r1"] # axis
            # old_protoParam["r2"] = new_proto_params["r2"] # axis
            # old_protoParam["cx"] = new_proto_params["cx"] # patch centers
            # old_protoParam["cy"] = new_proto_params["cy"] # --
            # # Rotated by theta
            # old_protoParam["theta"] = new_proto_params["theta"] # Normal form parameters


# function [numproto new_protoParam] = esSampleProtoParameters(M_tMap, old_protoParam)
# %esSampleProtoParameters - Generates the patch map M(t) parameters $$\theta_p$$
# %
# % Synopsis
# %          [numproto new_protoParam] = esSampleProtoParameters(M_tMap, old_protoParam)
# %
# % Description
# %     In a first step finds the boundaries of the actual patches
# %     In a second step get the N_V best patches ranked through their size and returns the actual
# %     M(t) map
# %
# %
# % Inputs ([]s are optional)
# %   (matrix) M_tMap            the patch map M(t)
# %   (struct) old_protoParam    the patch parameters at time step t-1
# %
# %
# % Outputs ([]s are optional)
# %   (integer) numproto         the actual number of patches
# %   (struct) new_protoParam    the patch parameters at current time step t:
# %                                the proto-objects boundaries: B{p}
# %                                 - new_protoParam.B
# %                                the proto-objects fitting ellipses parameters:
# %                                   a(1)x^2 + a(2)xy + a(3)y^2 + a(4)x + a(5)y + a(6) = 0
# %                                 - new_protoParam.a      conics parameters: a{p}
# %                                the normal form parameters: ((x-cx)/r1)^2 + ((y-cy)/r2)^2 = 1
# %                                 - new_protoParam.r1     normal form parameters
# %                                 - new_protoParam.r2     normal form parameters
# %                                 - new_protoParam.cx     normal form parameters
# %                                 - new_protoParam.cy     normal form parameters
# %                                the rotation parameter
# %                                 - new_protoParam.theta  normal form parameters

# % References
# %
# %    G. Boccignone and M. Ferraro, Ecological Sampling of Gaze Shifts,
# %                                     IEEE Trans. SMC-B, to appear
# %
# %    R. Hal?r and J. Flusser, Numerically stable direct least squares fitting of ellipses,
# %                             in Proc. Int. Conf. in Central Europe on Computer Graphics,
# %                             Visualization and Interactive Digital Media,
# %                             vol. 1, 1998, pp. 125?132.


    def _calculate_center_area(self, num_proto):
         # Determine the center and the area of patches for
         # subsequent IP sampling
        cx = self.cx
        cy = self.cy
        if num_proto > 0:
            proto_object_centers = np.array([list(cx.values()), list(cy.values())]).T
            nV = proto_object_centers.shape[0]
            logger.verbose(f"Number of protObject_centers: {proto_object_centers.shape[0]}")

            area_proto = np.zeros(nV)
            for p in range(0, nV):
                # for all proto-objects: area of the fitting ellipse/area of the saliency map
                area_proto[p] = self.r1[p]*self.r2[p]*np.pi

        self.nV = nV
        self.proto_centers = proto_object_centers
        self.area_proto = area_proto

    def _sample_proto_params(self, mt_map):
        # Computing patch boundaries
        feat_map_img = mt_map * 255
        _, thresh = cv2.threshold(cv2.convertScaleAbs(feat_map_img),0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        B = measure.find_contours(thresh, 0.5)
        # The actual patch number
        num_proto = len(B)

        if num_proto != 0:
            a = {}
            r1 = {}
            r2 = {}
            cx = {}
            cy = {}
            theta = {}

            for p in range(num_proto):
                boundary = np.flip(np.array(B[p]),1).astype(int)
                a[p] = cv2.fitEllipse(boundary)
                r1[p] = a[p][1][0] / 2.
                r2[p] = a[p][1][1] / 2.
                cx[p] = a[p][0][0]
                cy[p] = a[p][0][1]
                theta[p] = a[p][2]

            # Assign the new parameters
            self.B = B # The proto-objects boundaries: B{p}
            # The proto-objects fitting ellipses parameters:
            self.a = a # Conics parameters: a{p}
            self.r1 = r1 # Mormal form parameters
            self.r2 = r2 # Normal form parameters
            self.cx = cx # Normal form parameters
            self.cy = cy # Normal form parameters
            self.theta = theta # Normal form parameters
        else:
            logger.warning("No proto-objects found, keeping old ones")

        return num_proto

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

    def _sample_proto_map(self, s_map):
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

        proto_map_raw = np.zeros(s_map.shape)
        # Normalizing salience
        norm_sal = s_map
        max_sal = np.max(norm_sal)
        min_sal = np.min(norm_sal)
        norm_sal = np.divide((norm_sal-min_sal),(max_sal-min_sal))
        norm_sal = norm_sal*100

        # Method percentile based
        ind = np.stack(np.where(norm_sal >= np.percentile(norm_sal,95)), axis=-1)
        proto_map_raw[ind[:,0], ind[:,1]] = 1

        # Samples the N_V best patches
        opening_window=7
        M_tMap = self._sample_best_patches(proto_map_raw, opening_window)

        proto_map = np.logical_not(M_tMap)

        return M_tMap, proto_map, norm_sal


    def _sample_best_patches(self, proto_map_raw, win):
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

        contours, _ = cv2.findContours(cv2.convertScaleAbs(proto_map_raw),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        size_c = []
        for i,c in enumerate(contours):
            size_c.append(c.shape[0])
        sort_idx = np.argsort(np.array(size_c))[::-1]

        M_tMap = np.zeros(proto_map_raw.shape)
        nBest = min(self.n_best_proto, len(contours))

        for i in range(nBest):
            img = np.zeros(proto_map_raw.shape)
            cv2.fillPoly(img, pts =[contours[sort_idx[i]]], color=(255,255,255))
            img = img/np.max(img)
            M_tMap = M_tMap + img

        return M_tMap

    def sample_proto_objects(self, salience_map):

        # Using the proto-object representation which is the base of method
        # described in IEEE Trans SMC paper [2]
        #
        # If no proto-object are detected or PROTO is false, then we simply
        # go back to the original procedure described in the ICIAP 2011
        # paper [2]

        # Sampling the patch or proto-object map M(t)
        logger.verbose('Sampling the proto-object map')
        mt_map, proto_map, saliency_norm = self._sample_proto_map(salience_map)

        self.show_proto = np.ma.masked_where(proto_map == 0, proto_map)
        # We now have:
        #   the proto-object map                            M(t)
        #   the overlay rapresentation of proto-objects:    protoMap
        #   the normalized saliency:                        saliency_norm

        # Sampling the proto-object parameters
        logger.verbose('Sampling the proto-object parameters')
        num_proto = self._sample_proto_params(mt_map)
        self._calculate_center_area(num_proto)

        return num_proto
