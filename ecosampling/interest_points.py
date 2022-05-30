from utils.logger import Logger
from config import IPConfig
import numpy as np
import pymc3 as pm

from utils.discreteSampler import discrete_sampler



logger = Logger(__name__)

class IPSampler:

    def __init__(self):
        self.xCoord = None
        self.yCoord = None
        self.N = 0
        self.max_points = IPConfig.MAX_POINTS
        self.landscape_sampling = IPConfig.WITH_PERTURBATION
        self.num_samples = 0
        self.num_bins = 0
        self.show_hist = None


    def histogram_ips(self, frame_sampling, sampled_points_coord):
        # $$C(t)$$ captures the time-varying configurational complexity of interest points
        # within the landscape

        # Step 1. Computing the 2D histogram of IPs
        # Inputs:
        #     SampledPointsCoord: N x 2 real array containing N data points or N x 1 array
        #     nbinsx:             number of bins in the x dimension (defaults to 20)
        #     nbinsy:             number of bins in the y dimension (defaults to 20)
        logger.verbose("Histogramming interest points")

        n_bins_x = np.floor(frame_sampling.n_rows / IPConfig.X_BIN_SIZE).astype(int)
        n_bins_y = np.floor(frame_sampling.n_cols / IPConfig.Y_BIN_SIZE).astype(int)

        hist_mat, _, _ = np.histogram2d(sampled_points_coord[0], sampled_points_coord[1],
                                        bins=[n_bins_x, n_bins_y])

        #  We now have:
        #      histmat:   2D histogram array (rows represent X, columns represent Y)
        #      Xbins:     the X bin edges (see below)
        #      Ybins:     the Y bin edges (see below)
        num_samples = np.sum(np.sum(hist_mat))
        num_bins = hist_mat.shape[0]*hist_mat.shape[1]
        self.show_hist = hist_mat.T

        return hist_mat, num_samples, num_bins


    def interest_point_sample(self, num_proto, proto_params, saliency_map):
        # Sampling from proto-objects or directly from the map if numproto==0
        if num_proto > 0:
            self._proto_objects_sampling(proto_params)
            if self.landscape_sampling:
                self._landscape_sampling(saliency_map)
        else:
            logger.warn("No patches detected: Sampling interest points")
            self._landscape_sampling(saliency_map, only=True)

        return [self.xCoord, self.yCoord]


    def _proto_objects_sampling(self, proto_params):
        # Random sampling from proto-objects
        logger.verbose("Sample interest points from proto-objects")
        total_area = np.sum(proto_params.area_proto)

        all_points = np.empty((0, 2), float)

        for p in range(proto_params.nV):
            # Finds the number of IPs per patch
            n = round(3 * self.max_points * proto_params.area_proto[p] / total_area)
            if n > 0:
                self.N += n
                cov_proto = np.array([[(5*proto_params.r2[p]) , 0],
                                     [0, (5*proto_params.r1[p])]])
                mu_proto = proto_params.proto_centers[p]
                # PYMC
                mv_normal_dist = pm.MvNormal.dist(mu=mu_proto, cov=cov_proto, shape=(2, ))
                # print(.shape)

                r_p = mv_normal_dist.random(size=n)
                all_points = np.vstack((all_points, r_p))


        self.xCoord = all_points[:,0]
        self.yCoord = all_points[:,1]

    def _landscape_sampling(self, saliency_map, only=False):
        xCoord, yCoord, scale = self._boccignone_ferraro_ip_sampling(saliency_map)
        N = len(scale) # Number of points
        if not only:
            self.N += N
            self.xCoord = np.append(self.xCoord, xCoord)
            self.yCoord = np.append(self.yCoord, yCoord)
        else:
            self.N = N
            self.xCoord = xCoord
            self.yCoord = yCoord



# %InterestPoint_Sampling - Simple interest point generator
# %
# % Synopsis
# %   [xCord yCord scale ] = InterestPoint_Sampling(map,Interest_Point)
# %
# % Description
# %       1. Sample Interest_Point.Max_Points points from set of points, weighted according to their salience
# %       2. For each sample, set scale by drawing from uniform distribution ...
# %          over Interest_Point.Scale
# %
# % Inputs ([]s are optional)
# %   (matrix) map              Frame saliency map
# %   (struct) Interest_Point   structure holding all settings of the interest operator
# %   - (bool)Weighted_Sampling if true, using weighted sampling; otherwise,
# %                             uniform sampling
# %   - (int ) Max_Points       maximum number of points to sample
# %   - (bool)Weighted_Scale    if true, using weighted scale;
# %
# % Outputs ([]s are optional)
# %   (vector) xCord            (1 x Interest_Point.Max_Points) coordinates and scale of IPs
# %   (vector) yCord
# %   (vector) scale            (1 x Interest_Point.Max_Points) characteristic scale of points (radius,inpixels)
# %
# %
# % Authors
# %   Giuseppe Boccignone <Giuseppe.Boccignone(at)unimi.it>
# %
# % Changes
# %   12/12/2012  First Edition
# %


    def _boccignone_ferraro_ip_sampling(self, saliency_map):
        x = []
        xx = []
        y = []
        yy = []
        strength = []
        scale = []

        yy, xx, strength = self._get_points(saliency_map)
        # Total nomber of salient points extracted from image
        n_sal_points = len(strength)

        # Check that some points were found in the image
        if n_sal_points > 0:
            # Obtain sampling density choose between uniform
            # and weighted towards those points with a stronger saliency strength
            if IPConfig.WEIGHTED_SAMPLING:
                sample_density = strength / np.sum(strength)
            else:
                sample_density = np.ones((1, IPConfig.N_POINTS)) / IPConfig.N_POINTS

            # Choose how many points to sample
            n_points_to_sample = IPConfig.MAX_POINTS
            sample_density = sample_density.astype('double')

            # Draw samples from density
            samples = discrete_sampler(sample_density, n_points_to_sample)

            # Lookup points corresponding to samples
            x = xx[samples]
            y = yy[samples]

            # Now draw scales from uniform
            ip_scale = IPConfig.SCALE
            scale = np.random.rand(1, n_points_to_sample) * (np.max(ip_scale)-np.min(ip_scale)) + np.min(ip_scale)
        else:
            # No salient points found in image at all
            # Set all output variables for the frame to be empty
            x = []
            y = []
            scale = []

        xCoord = x
        yCoord = y

        return xCoord, yCoord, scale


    def _get_points(self, salience_map):
        mean_salience = np.mean(np.mean(salience_map))
        indexes = np.argwhere(salience_map > mean_salience)
        xx = indexes[:, 0]
        yy = indexes[:, 1]
        strength = salience_map[xx, yy]

        return yy, xx, strength


