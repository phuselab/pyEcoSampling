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
# %
# % Requirements
# %   fitellip.m
# %
# % References
# %
# %    G. Boccignone and M. Ferraro, Ecological Sampling of Gaze Shifts,
# %                                     IEEE Trans. SMC-B, to appear
# %
# %    R. Hal?r and J. Flusser, Numerically stable direct least squares fitting of ellipses,
# %                             in Proc. Int. Conf. in Central Europe on Computer Graphics,
# %                             Visualization and Interactive Digital Media,
# %                             vol. 1, 1998, pp. 125?132.
# %
# %
# %
# % Author
# %   Giuseppe Boccignone <Giuseppe.Boccignone(at)unimi.it>
# %
# %
# % Changes
# %   12/12/2012  First Edition
# %

import cv2
from skimage import measure
import numpy as np

def esSampleProtoParameters(mt_map, old_proto_params):
    # Computing patch boundaries
    feat_map_img = mt_map*255
    _, thresh = cv2.threshold(cv2.convertScaleAbs(feat_map_img),0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    B = measure.find_contours(thresh, 0.5)
    # The actual patch number
    num_proto = len(B)
    new_proto_params = {}
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
        new_proto_params["B"] = B # The proto-objects boundaries: B{p}
        # The proto-objects fitting ellipses parameters:
        new_proto_params["a"] = a # Conics parameters: a{p}
        new_proto_params["r1"] = r1 # Mormal form parameters
        new_proto_params["r2"] = r2 # Normal form parameters
        new_proto_params["cx"] = cx # Normal form parameters
        new_proto_params["cy"] = cy # Normal form parameters
        new_proto_params["theta"] = theta # Normal form parameters
    else:
        # Use the old ones
        new_proto_params = old_proto_params

    return num_proto, new_proto_params






















		# if self.name == 'STS':
		# 	nBestProto = 10
		# 	M_tMap, protoMap = self.esSampleProtoMap(nBestProto)

		# else:
		# 	feat_map_img = (self.feat_map/np.max(self.feat_map))*255

		# ret,thresh = cv2.threshold(cv2.convertScaleAbs(feat_map_img),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		# #B,hierarchy = cv2.findContours(cv2.convertScaleAbs(thresh), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# B = measure.find_contours(thresh, 0.5)

		# numproto = len(B)
		# #bX = {}
		# #bY = {}
		# a = {}
		# r1 = {}
		# r2 = {}
		# cx = {}
		# cy = {}
		# theta = {}
		# boundaries = {}

		# for p in range(numproto):
		# 	#boundary = np.squeeze(B[p])	#for cv2
		# 	boundary = np.flip(np.array(B[p]),1).astype(int)
		# 	#boundary = np.array(B[p]).astype(int)
		# 	boundaries[p] = boundary
		# 	#bX[p] = boundary[:,0]
		# 	#bY[p] = boundary[:,1]

		# 	a[p] = cv2.fitEllipse(boundary)

		# 	r1[p] = a[p][1][0] / 2.
		# 	r2[p] = a[p][1][1] / 2.
		# 	cx[p] = a[p][0][0]
		# 	cy[p] = a[p][0][1]
		# 	theta[p] = a[p][2]

		# self.numproto = numproto
		# self.boundaries = boundaries
		# self.ellipse = a
		# self.radius1 = r1
		# self.radius2 = r2
		# self.centerx = cx
		# self.centery = cy
		# self.theta = theta
