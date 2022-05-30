
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

import numpy as np
from config import IPConfig
from utils.discreteSampler import discrete_sampler


def interest_point_sampling(saliency_map):
    x = []
    xx = []
    y = []
    yy = []
    strength = []
    scale = []

    yy, xx, strength = get_points(saliency_map)
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


def get_points(salience_map):
    mean_salience = np.mean(np.mean(salience_map))
    indexes = np.argwhere(salience_map > mean_salience)
    xx = indexes[:, 0]
    yy = indexes[:, 1]
    strength = salience_map[xx, yy]

    return yy, xx, strength


