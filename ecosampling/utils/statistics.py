"""Collection of statistics related functions.

Authors:
    - Giuseppe Boccignone <giuseppe.boccignone@unimi.it>
    - Renato Nobre <renato.avellarnobre@studenti.unimi.it>

Changes:
    - 12/12/2012  First Edition Matlab
    - 31/05/2022  Python Edition
"""

from re import S
import numpy as np

from scipy.stats import multivariate_normal
from scipy.stats import dirichlet
from scipy.stats import multinomial
from scipy.stats import levy_stable

def discrete_sampler(density, num_samples, replacement_option=True):
    """Function that draws samples from a discrete density.

    Args:
        density (vector): discrete probability density (should sum to 1)
        num_samples (int): number of samples to draw
        replacement_option (bool, optional): True for sampling with replacement
            False for non replacement. Defaults to True.

    Returns:
        Samples drown from the discrete density.
    """
    samples_out = np.zeros((1, num_samples))

    # Get CDF
    cdf = np.cumsum(density)

    # Draw samples from Uniform Distribution
    uniform_samples = np.random.rand(num_samples)

    a = 0
    while a <= num_samples-1:
        binary = uniform_samples[a] > cdf
        highest = np.argwhere(binary)

        if highest.size == 0:
            samples_out[0, a] = 1
        else:
            samples_out[0, a] = highest[-1] + 1

        # If we aren't doing replacement
        if (not replacement_option and a>0):
            if (np.sum(samples_out[0, a] == samples_out[0, 0:a]) > 0):
                uniform_samples[0, a] = np.random.rand(1)[0]; # Gen. new uniform sample
                a -= 1 # Redo this sample

        a += 1

    return np.squeeze(samples_out).astype(int)


def sample_discrete(prob, n=1):
    """Draw from a non-uniform discrete distribution

    Multinomial.

    Example:

        sample_discrete([0.8, 0.2], 1, 10) generates a row vector of 10 random integers from {1,2},
        where the prob. of being 1 is 0.8 and the prob of being 2 is 0.2.

    Args:
        prob (np.array): Vector of probabilities.
        r (int, optional): Starting value of the random integer.
            Defaults to 1.

    Returns:
       Sampled data.
    """
    sampled_values = multinomial.rvs(n, prob[0])
    return np.where(sampled_values == 1)[0]

def sample_dirchlet(nu, size):
    """Sample from a Dirchlet distribution.

    Args:
        nu (np.array): Concentration parameters.
        size (int): Amount of samples to draw.

    Returns:
        Drawn samples from a Dirchlet distribution.
    """
    sampled_values = dirichlet.rvs(nu, size=size)
    return sampled_values

def sample_multivariate(mu, cov, sample_size):
    """Sample from a multivariate distribution.

    Args:
        mu (np.ndarray): Vector of means
        cov (np.ndarray): Covariance matrix.
        shape (tuple): Shape of the samples.
        sample_size (int): Amount of samples to draw.

    Returns:
        Samples drawn from a multivariate normal distribution.
    """
    sampled_values = multivariate_normal.rvs(mean=mu, cov=cov, size=sample_size)
    return sampled_values

def sample_levy_stable(alpha, beta, scale, loc, size):
    """Sample from a Levy-stable distribution.

    Args:
        alpha (float): Stability parameter.
        beta (float): Symmetry parameter.
        scale (float): Scale parameter.
        loc (float): Location parameter.
        size (int): Amount of samples to draw.

    Returns:
        Sample drawn from a Levy-stable distribution.
    """
    samples = levy_stable.rvs(alpha, beta, scale=scale, loc=loc, size=size)
    return samples
