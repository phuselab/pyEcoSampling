"""Generate Interest Points and captures complexity within the landscape.

Sample points from set of points, weighted according to their salience and
captures the time-varying configurational complexity of interest points
within the landscape, generating a 2D Histogram of IPs.

Authors:
    - Giuseppe Boccignone <giuseppe.boccignone@unimi.it>
    - Renato Nobre <renato.avellarnobre@studenti.unimi.it>

Changes:
    - 12/12/2012  First Edition Matlab
    - 31/05/2022  Python Edition
"""

import numpy as np
import pymc3 as pm

from config import IPConfig
from utils.statistics import discrete_sampler
from utils.logger import Logger

logger = Logger(__name__)

class IPSampler:
    """Generate Interest Points and captures complexity within the landscape.

    Attributes:
        xCoord (np.array): X Coordinates of sampled points.
        yCoord (np.array): Y Coordinates of sampled points.
        show_hist (np.array): 2D histogram of IPs for visualization.
        N (int): Number of sampled points.
        num_samples (int): Number of samples in the 2D histogram.
        num_bins (int): Number of bins in the 2D histogram.
        max_points (int): Maximum number of points to sample. Value is set in
            `config.py`
        landscape_sampling (bool): Landscape sampligling flag. If True,
            sample also from the landscape. Value is set in `config.py`
    """

    def __init__(self):
        self.xCoord = None
        self.yCoord = None
        self.show_hist = None
        self.N = 0
        self.num_samples = 0
        self.num_bins = 0
        self.max_points = IPConfig.MAX_POINTS
        self.landscape_sampling = IPConfig.WITH_PERTURBATION

    def interest_point_sample(self, num_proto, proto_params, saliency_map):
        """Sample Interest points from proto-objects and/or from the landscape.

        Control the sampling of interest points. If there is no proto-objects,
        sample only from the landscape. If there are proto-objects, sample from both
        if `self.landscape_sampling` is True, otherwise only from proto-objects.

        Args:
            num_proto (int): Ammount of proto-objects.
            proto_params (obj): ProtoParameters object.
            saliency_map (np.ndarray): Frame saliency map.

        Returns:
            Vector of XCoord and YCoord
        """
        # Sampling from proto-object
        if num_proto > 0:
            self._proto_objects_sampling(proto_params)
            if self.landscape_sampling:
                self._landscape_sampling(saliency_map)
        # Sampling from the map if there are no proto-objects
        else:
            logger.warn("No patches detected: Sampling interest points")
            self._landscape_sampling(saliency_map, only=True)

        return [self.xCoord, self.yCoord]

    def histogram_ips(self, frame_sampling, sampled_points_coord):
        """Time-varying configurational complexity of interest points.

        :math:`C(t)` captures the time-varying configurational complexity of
        interest points within the landscape

        Args:
            sampled_points_coord (np.array): (N, 2) array containing data points coordinated.

        Returns:
            histmat (nd.array): 2D histogram array (rows represent X, columns represent Y)
            num_samples (int): Number of sample points in the 2D histogram
            num_bins (int): Number of bins in the 2D histogram
        """
        logger.verbose("Histogramming interest points")

        n_bins_x = np.floor(frame_sampling.n_rows / IPConfig.X_BIN_SIZE).astype(int)
        n_bins_y = np.floor(frame_sampling.n_cols / IPConfig.Y_BIN_SIZE).astype(int)
        # Step 1. Computing the 2D histogram of IPs
        hist_mat, _, _ = np.histogram2d(sampled_points_coord[0], sampled_points_coord[1],
                                        bins=[n_bins_x, n_bins_y])

        num_samples = np.sum(np.sum(hist_mat))
        num_bins = hist_mat.shape[0]*hist_mat.shape[1]
        self.show_hist = hist_mat.T

        return hist_mat, num_samples, num_bins


    def _proto_objects_sampling(self, proto_params):
        """Sample points from the proto-objects.

        Args:
            proto_params (obj): Sampled proto parameters for the frame

        Note:
            Update `self.xCoord` and `self.yCoord` with the sampled points
        """
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
        """Sample points from the landscape.

        Note:
            Update `self.xCoord` and `self.yCoord` with the sampled points

        Args:
            saliency_map (np.ndarray): Frame saliency map
            only (bool, optional): Flag indicating if we are only sampling
                landscape or if the data will be appended to other
                previously sampled points. Defaults to False.

        """
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


    def _boccignone_ferraro_ip_sampling(self, saliency_map):
        """Boccignone-Ferraro's backend function for the IP sampling method.

        1. Sample Interest_Point.Max_Points points from set of points
           weighted according to their salience
        2. For each sample, set scale by drawing from uniform distribution
           over Interest_Point.Scale

        Args:
            saliency_map (np.ndarray): Frame saliency map

        Returns:
            xCoord (np.array): X Coordinates of IPs
            yCoord (np.array): Y Coordinates of IPs
            scale (np.array): Scale of points (radius, inpixels)
        """

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

            # Draw samples from density
            sample_density = sample_density.astype('double')
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
        """Extract salient points from the saliency map.

        Args:
            saliency_map (np.ndarray): Frame saliency map

        Returns:
            yy: Y coordinates of salient points
            xx: X coordinates of salient points
            strength: Saliency strength of salient points
        """
        mean_salience = np.mean(np.mean(salience_map))
        indexes = np.argwhere(salience_map > mean_salience)
        xx = indexes[:, 0]
        yy = indexes[:, 1]
        strength = salience_map[xx, yy]

        return yy, xx, strength


