"""Handle creation of the proto-objects its visualizations.

Sample the patch or proto-object map M(t), the proto parameters and
calculate the center of mass of the proto-objects. Additionally, creates
the visual representation of the proto-objects.

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
    """Handle creation of the proto-objects its visualizations.

    Note:
        The proto-parameters, its visual representation, centers, and areas
        are stored in the object, and are updated at each frame.

    Attributes:
        B (np.ndarray): Patch boundaries
        a (np.ndarray): The proto-objects fitting ellipses parameters
        r1 (np.ndarray): Normal form parameters axis
        r2 (np.ndarray): Normal form parameters axis
        cx (np.ndarray): Normal form center X
        cy (np.ndarray): Normal form center Y
        theta (np.ndarray): Elipse Rotation
        show_proto (np.ndarray): Visual representation of proto objects.
        n_best_proto (int): Maximum ammount of best proto objects to sample.
        area_proto (np.ndarray): Area of proto objects ellipses.
        nV (int): Number of vertices.
        proto_centers (np.ndarray): Centers of proto objects.
    """

    def __init__(self):
        self.B = []
        self.a = []
        self.r1, self.r2 = [], []
        self.cx, self.cy = [], []
        self.theta = []
        self.nV = 0
        self.show_proto = None
        self.area_proto = None
        self.proto_centers = None
        self.n_best_proto = ProtoConfig.N_BEST_PROTO

    def sample_proto_objects(self, salience_map):
        """Sample the proto-objects and create visualizations.

        Sample the patch or proto-object map M(t), the proto parameters and
        calculate the center of mass of the proto-objects. Additionally, creates
        the visual representation of the proto-objects.

        Using the proto-object representation which is the base of method
        described in [1].

        Note:
            The proto-parameters and its visual representation are stored in the object,
            and are updated at each frame.

        Args:
            saliency_map (np.ndarray): Frame saliency map

        Returns:
            num_proto (int): Number of proto-objects found.

        References
        ----------
        .. [1] `Halir, R., & Flusser, J. (1998, February). Numerically stable direct least squares
           fitting of ellipses. In Proc. 6th International Conference in Central Europe
           on Computer Graphics and Visualization. WSCG (Vol. 98, pp. 125-132). Citeseer.
           <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1.7559&rep=rep1&type=pdf>`_
        """
        # Sampling the patch or proto-object map M(t)
        logger.verbose('Sampling the proto-object map')
        mt_map, proto_map, saliency_norm = self._sample_proto_map(salience_map)

        # Create show version of proto-objects
        self.show_proto = np.ma.masked_where(proto_map == 0, proto_map)

        # Sampling the proto-object parameters
        logger.verbose('Sampling the proto-object parameters')
        num_proto = self._sample_proto_params(mt_map)

        # Calculate centers
        self._calculate_center_area(num_proto)

        return num_proto


    def _calculate_center_area(self, num_proto):
        """Calculate center and Area of proto-objects.

        Determine the center and the area of patches for
        subsequent IP sampling.

        Note:
            The proto-parameters area and centers are stored in the object,
            and are updated at each frame.

        Args:
            num_proto (int): Number of proto-objects found.
        """
        cx = self.cx
        cy = self.cy
        if num_proto > 0:
            proto_object_centers = np.array([list(cx.values()), list(cy.values())]).T
            nV = proto_object_centers.shape[0]
            logger.verbose(f"Number of proto_object_centers: {proto_object_centers.shape[0]}")

            area_proto = np.zeros(nV)
            for p in range(0, nV):
                # Aea of the fitting ellipse/area of the saliency map
                area_proto[p] = self.r1[p]*self.r2[p]*np.pi

        self.nV = nV
        self.proto_centers = proto_object_centers
        self.area_proto = area_proto


    def _sample_proto_params(self, mt_map):
        """Update the class patch map M(t) parameters :math:`theta_p`.

        In a first step finds the boundaries of the actual patches.

        Note:
            The proto-parameters and are stored in the object,
            and are updated at each frame.

        Args:
            mt_map (np.ndarray): the patch map :math:`M(t)`

        Returns:
            num_proto (int): Number of protoparametes
        """
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

            self.B = B
            # The proto-objects fitting ellipses parameters:
            self.a = a
            self.r1 = r1
            self.r2 = r2
            self.cx = cx
            self.cy = cy
            self.theta = theta
        else:
            logger.warning("No proto-objects found, keeping old ones")

        return num_proto

    def _sample_proto_map(self, s_map):
        """Generates the patch map M(t).

        In a first step generates the raw patch map by thresholding the normalized salience map
        so as to achieve 95% significance level for deciding whether the given saliency values are
        in the extreme tails.

        Args:
            s_map (matrix): the salience map, 0/1 overlay representation
            curr_frame (matrix): the current frame
            n_best_proto (integer): the N_V most valuable patches

        Returns:
            mt_map (matrix): the patch map M(t)
            proto_map (matrix): the object layer representation of patch map M(t)
            norm_sal (matrix): the normalized salience map

        References
        ----------
        .. [1] `Boccignone, G., & Ferraro, M. (2013). Ecological sampling of gaze shifts.
           IEEE transactions on cybernetics, 44(2), 266-279.
           <https://ieeexplore.ieee.org/abstract/document/6502674>`_
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
        mt_map = self._sample_best_patches(proto_map_raw, opening_window)

        proto_map = np.logical_not(mt_map)

        return mt_map, proto_map, norm_sal


    def _sample_best_patches(self, proto_map_raw):
        """Samples the N_V best patches.

        Samples the N_V best patches ranked through their size and returns the actual
        M(t) map.

        Args:
            proto_map_raw (matrix): the raw patch map

        Returns:
            mt_map (matrix): the patch map M(t)
        """
        contours, _ = cv2.findContours(cv2.convertScaleAbs(proto_map_raw),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        size_c = []
        for i,c in enumerate(contours):
            size_c.append(c.shape[0])
        sort_idx = np.argsort(np.array(size_c))[::-1]

        mt_map = np.zeros(proto_map_raw.shape)
        nBest = min(self.n_best_proto, len(contours))

        for i in range(nBest):
            img = np.zeros(proto_map_raw.shape)
            cv2.fillPoly(img, pts =[contours[sort_idx[i]]], color=(255,255,255))
            img = img/np.max(img)
            mt_map = mt_map + img

        return mt_map
