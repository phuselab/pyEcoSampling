"""File for the GazeSampler class.

For further information, see also [1]_ and [2]_.

Authors:
    - Giuseppe Boccignone <giuseppe.boccignone@unimi.it>
    - Renato Nobre <renato.avellarnobre@studenti.unimi.it>

Changes:
    - 12/12/2012  First Edition Matlab
    - 31/05/2022  Python Edition

References
----------
.. [1] `Boccignone, G., & Ferraro, M. (2013). Ecological sampling of gaze shifts.
    IEEE transactions on cybernetics, 44(2), 266-279.
    <https://ieeexplore.ieee.org/abstract/document/6502674>`_
.. [2] `G. Boccignone and M. Ferraro, The active sampling of gaze-shifts,
    in Image Analysis and Processing ICIAP 2011, ser. Lecture Notes in Computer Science,
    G. Maino and G. Foresti, Eds. Springer Berlin / Heidelberg, 2011, vol. 6978, pp. 187?196.
    <https://ieeexplore.ieee.org/abstract/document/6502674>`_
"""

import numpy as np
from config import GazeConfig, IPConfig
from utils.logger import Logger
from scipy.spatial import KDTree, cKDTree
from utils.statistics import sample_multivariate, sample_levy_stable

logger = Logger(__name__)

class GazeSampler:
    """Compute the future Focus of Attention, using gaze sampling.

    Function computing, using gaze attractors, the actual gaze relocation by
    sampling the appropriate noise parameters from a mixture of alpha-stable distributions
    as a function of the oculomotor state, The parameters are used to propose one or multiple
    candidate gaze shifts actually implemented by calling for ``_langevin_sampling``,
    which performs the Langevin step. Eventually decides and sets the actual shift
    to the new Focus of Attention.

    Attributes:
        start_nu (np.array): vector of the starting probabilities for the Dirichlet distribution.
        start_foa (np.array): vector of the starting FOA coordinates.
        num_internalsim (int): maximum allowed number of candidate new  gaze position r_new
        max_numattempts (int): maximum number of attempts to sample a new valid gaze position.
        simple_attractors (bool): if True, the attractors are set to the FOA.
        alpha (float): alpha parameter of the stable distribution.
        beta (float): beta parameter of the stable distribution.
        gamma (float): gamma parameter of the stable distribution.
        delta (float): delta parameter of the stable distribution.
        foa_size (int): size of the FOA.
        frame_sampling (FrameSampling): frame sampling object.
        ip_sampler (IPSampler): ip sampling object.
        num_proto (int): number of prototypes.
        landscape (dict): The parameters for setting the landscape representation.
        candidate_foa (np.array): candidate FOA coordinates.
        candx (np.array): candidate FOA x coordinates.
        candy (np.array): candidate FOA y coordinates.
        show (np.ndarray): visualization mask of the FOA.

    Parameters:
        frame_sampling (FrameSampling): frame sampling object
        proto_params (ProtoParams): proto params object
        ip_sampler (IPSampler): ip sampler object
        hist_mat (np.array): histogram matrix
        num_proto (int): number of proto-objects

    Raises:
        NotImplementedError: No methods defined for setting the first FOA
    """

    def __init__(self, frame_sampling, proto_params=None, ip_sampler=None, hist_mat=None, num_proto=0):

        # Starting Values
        self.start_nu = np.ones(3)
        if GazeConfig.FIRST_FOA_ON_CENTER:
            x_center = round(frame_sampling.n_rows/2)
            y_center = round(frame_sampling.n_cols/2)
        else:
            raise NotImplementedError('No methods defined for setting the first FOA')

        # Instance values
        self.start_foa = np.array([x_center, y_center])
        self.num_internalsim = GazeConfig.NUM_INTERNALSIM
        self.max_numattempts = GazeConfig.MAX_NUMATTEMPTS
        self.simple_attractors = GazeConfig.SIMPLE_ATTRACTOR
        self.alpha = GazeConfig.ALPHA_STABLE
        self.beta = GazeConfig.BETA_STABLE
        self.gamma = GazeConfig.GAMMA_STABLE
        self.delta = GazeConfig.DELTA_STABLE
        self.foa_size = round(max(frame_sampling.n_rows,
                                  frame_sampling.n_cols) / 6)
        self.frame_sampling = frame_sampling
        self.ip_sampler = ip_sampler
        self.num_proto = num_proto

        self.candidate_foa = None
        self.candx = None
        self.candy = None
        self.show = None

        # Landscape Configuration
        self.landscape = {}
        if self.num_proto > 0:
            self.landscape["area_proto"] = proto_params.area_proto
            self.landscape["proto_centers"] = proto_params.proto_centers
        else:
            self.landscape["histmat"] = hist_mat
            self.landscape["xbinsize"] = IPConfig.X_BIN_SIZE
            self.landscape["ybinsize"] = IPConfig.Y_BIN_SIZE
            self.landscape["NMAX"] = GazeConfig.NMAX


    def sample_gaze_shift(self, z, pred_foa, dir_old):
        """Sampling the gaze shift.

        Setting the attractors of the FOA, sampling the FOA, which is
        returned together with the simulated candidates
        setting the oculomotor state z parameter

        Args:
            z (int): Index of kind of gaze shift
            pred_foa (np.ndarray): Previous FOA coordinates
            dir_old (float): Value of the previous direction

        Returns:
            final_foa (np.ndarray): Final FOA coordinates
            dir_new (float): Value of the new direction
        """
        logger.verbose("Sample gaze point")
        z = z.squeeze().astype(int)


        foa_attractors = self._get_gaze_attractors(pred_foa)
        logger.verbose(f"FOA attractors: {foa_attractors}")
        final_foa, dir_new = self._gaze_sampling(z, foa_attractors, pred_foa, dir_old)

        self.show = self._create_circular_mask(final_foa)

        return final_foa, dir_new



    def _get_gaze_attractors(self, pred_foa):
        """Samples the possible gaze attractors.

        Function computing possible ONE or MULTIPLE gaze attractors
        If a landscape of proto-objects is given then their centers are used as described in [1].
        Otherwise attractors are determined through the IPs sampled from saliency as in [2].

        Args:
            pred_foa (np.ndarray): (1, 2) vector representing the previous FoA coordinatesxs

        Returns:
            foa_attractors (np.ndarray): (N_V, 2) matrix representing the FoA attractors
        """
        # If true: if multiple maxima, choose the first closest one
        # to the previous FOA for stability purposes
        MAKE_STABLE = False

        # Setting the landscape
        if self.num_proto > 0:
            area_proto = self.landscape["area_proto"]
            prot_object_centers = self.landscape["proto_centers"]
        else:
            histmat = self.landscape["histmat"]
            xbin_size = self.landscape["xbinsize"]
            ybin_size = self.landscape["ybinsize"]
            NMAX = self.landscape["NMAX"]

        # Landscape Evaluation
        if self.simple_attractors:
            if self.num_proto > 0:
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

                    kdt = KDTree(X.T)
                    _, k = kdt.query(XI.T)

                    # This actually is the column index in the image bitmap
                    foa_x = round(x_max[k, 0]*xbin_size - xbin_size/2)
                    # This actually is the row index in the image bitmap!
                    foa_y = round(y_max[k, 0]*ybin_size - ybin_size/2)

                    # Swap image coordinates
                    foa_x, foa_y = foa_y, foa_x

            # Now we have at least 1 potential candidate FOA
            # simple one point attractor: use the candidate FOA
            foa_attractors = np.array([foa_x, foa_y])
        else:
            # Multipoint attractor
            if self.num_proto > 0:
                foa_attractors = np.zeros(prot_object_centers.shape)
                foa_attractors[:,0] = prot_object_centers[:,0].round()
                foa_attractors[:,1] = prot_object_centers[:,1].round()
            else:
                # find first  NMAX to determine the total attractor potential in LANGEVIN:
                ms = np.sort(histmat, axis=None)[::-1]
                rx, cx = np.unravel_index(np.argsort(histmat, axis=None), histmat.shape)

                # row col val: row col inverted with respect to image coord
                foa_attractors_all = np.concatenate((rx[:,None], cx[:,None], ms[:,None]), axis=1)
                foa_attractors_all = foa_attractors_all[0:NMAX,:]
                # Retains only row col
                foa_attractors = foa_attractors_all[:,0:2]
                # This actually is the column index in the image bitmap
                foa_attractors[:,0] = (foa_attractors[:,0]*xbin_size - xbin_size/2).round()
                # This actually is the column index in the image bitmap!
                foa_attractors[:,1] = (foa_attractors[:,1]*ybin_size - ybin_size/2).round()

                # Swap image coordinates
                foa_attractors = np.flip(foa_attractors, axis=1)

        return foa_attractors


    def _gaze_sampling(self, z, foa_attractors, pred_foa, dir_old):
        """Samples the new gaze position

        Function computing the actual gaze relocation by using a
        Langevin like stochastic differential equation,
        whose noise source is sampled from a mixture of alpha-stable distributions.
        By using NUM_INTERNALSIM generates an equal number of candidate
        parameters, that is of candidate new possible shifts which are then
        passed to the _langevin_sampling() function, which executes the
        Langevin step

        Note:
            If a valid new gaze point is returned this is set as the final FOA
            Otherwise, a gaussian noise "perturbated" candidate FOA is tried.
            The dafault solution, if all goes wrong, is to keep the old FOA

        Args:
            z (int): Kind of gaze shift
            foa_attractors (np.ndarray): (N_V, 2) matrix representing the FoA attractors
            pred_foa (np.ndarray): (1, 2) vector representing the previous FoA coordinatesxs
            dir_old (float): Value of the previous direction

        Returns:
            final_foa (np.ndarray): (1, 2) vector representing the new FoA coordinates
            dir_new (float): Value of the new direction
        """
        nrow = self.frame_sampling.n_rows
        ncol = self.frame_sampling.n_cols

        s = self.foa_size / 4

        # Direction Sampling - Uniform random sampling
        dir_new = 2*np.pi*np.random.rand(1, self.num_internalsim)

        # Shaping the potential of Langevin equation
        if self.simple_attractors:
            candidate_foa = foa_attractors.round()
            dhx = -(pred_foa[0]-foa_attractors[0])
            dhy = -(pred_foa[1]-foa_attractors[1])
        else:
            # Set the center of mass of attractors as a potential candidate FOA
            candidate_foa = (np.sum(foa_attractors, axis=0)/foa_attractors.shape[0]).round()
            dhx = -(pred_foa[0]-foa_attractors[:,0])
            dhy = -(pred_foa[1]-foa_attractors[:,1])
        T = 1 # Maximum time
        N = 30

        sde_param = {}
        # SDE new direction of the gaze shift
        sde_param["dir_new"] = dir_new
        # SDE gradient potential x and y coordinate
        sde_param["dHx"] = np.sum(dhx)
        sde_param["dHy"] = np.sum(dhy)
        # SDE integration Time step
        sde_param["h"] = T / N
        # SDE alpha-stable characteristic exponent parameter
        sde_param["alpha"] = self.alpha[z]
        # SDE alpha-stable scale parameter
        sde_param["gamma"] = self.gamma[z]

        accept = 0
        count_attempts = 0

        # Cycling until a sampled FOA is accepted
        while (not accept) and (count_attempts < self.max_numattempts):
            # setting alpha-stable distribution parameters according to the
            # regime specified by rv z
            xi = sample_levy_stable(self.alpha[z], self.beta[z],
                                    scale=self.gamma[z], loc=self.delta[z],
                                    size=self.num_internalsim)

            # Setting Langevin SDE parameters
            # SDE alpha-stable component
            sde_param["xi"] = xi

            # Langevin gaze shift sampling
            # final_foa is the sampled new Gaze position
            final_foa, dir_new, accept = self._langevin_sampling(sde_param, pred_foa)
            count_attempts += 1
            logger.verbose(f"Trying...count_attempts = {count_attempts}")

        # if something didn't work for some reason use a perturbed argmax solution
        if not accept:
            # For normal regime simple choice on most salient point
            new = sample_multivariate(candidate_foa.conj().T, np.eye(2)*s, 1)
            new = new.conj().T
            validcord = np.nonzero((new[0] >= 1) & (new[0] < nrow) &
                                   (new[1] >= 1) & (new[1] < ncol))[0]

            logger.verbose(f'Sampled {validcord.size} valid candidate FOAs from normal')

            if validcord.size > 0:
                final_foa = new
                accept = 1
                logger.verbose('DEFAULT SOLUTION: Perturbed ArgMax solution!!')

        # The last chance: use the previous FOA
        if not accept:
            final_foa = pred_foa
            accept = 1
            logger.warn('BACKUP SOLUTION!!!!!! Keeping OLD FOA')

        self.candidate_foa = candidate_foa

        logger.verbose(f"Current FOA {pred_foa[0]}, {pred_foa[1]}")
        logger.verbose(f"Candidate Max FOA {candidate_foa[0]}, {candidate_foa[1]}")
        logger.verbose(f"Old dir {np.rad2deg(dir_old)}")
        logger.verbose(f"New dir {np.rad2deg(dir_new)}")
        logger.verbose(f"Final FOA {final_foa[0]}, {final_foa[1]}")
        dis = np.sqrt(np.square(final_foa[0]- candidate_foa[0]) + np.square(final_foa[1]- candidate_foa[1]))
        logger.verbose(f'Distance from candidate FOA {dis}')
        logger.verbose('-----FOA SAMPLING TERMINATED \n')

        return final_foa, dir_new

    def _langevin_sampling(self, sde_param, pred_foa):
        """Langevin step for sampling the new gaze position.

        Implements a step of the Langevin like stochastic
        differential equation (SDE), whose noise source is sampled
        from a mixture of alpha-stable distributions.

        Args:
            sde_param (dict): Langevin SDE parameters::

                {
                    "dHx": SDE gradient potential x coordinate,
                    "dHy": SDE gradient potential y coordinate,
                    "xi": SDE alpha-stable component
                    "dir_new": SDE new direction of the gaze shift
                    "alpha": SDE alpha-stable characteristic exponent parameter
                    "gamma": SDE alpha-stable scale parameter
                    "h": SDE integration step
                }

            pred_foa (np.ndarray): Previous FOA coordinates

        Returns:
            final_foa (np.ndarray): (1, 2) vector representing the new FoA coordinates
            dir_new (float): Value of the new direction
            accept (bool): 1 if the new FOA is accepted, 0 otherwise
        """
        # Rough initialization: nothing changes
        final_foa = pred_foa

        xCordIP = self.ip_sampler.xCoord
        yCordIP = self.ip_sampler.yCoord
        nrow = self.frame_sampling.n_rows
        ncol = self.frame_sampling.n_cols

        candx = []
        candy = []
        accept = 0

        dHx = sde_param["dHx"]
        dHy = sde_param["dHy"]
        xi = sde_param["xi"]
        dir_new = sde_param["dir_new"]
        alpha_stable = sde_param["alpha"]
        gamma_stable = sde_param["gamma"]
        h = sde_param["h"]

        # Sampling the shift dx, dy on the basis of the generalized discretized Langevin
        value = np.sqrt(gamma_stable)*(h**(1/alpha_stable))*xi
        dx = dHx*h + np.multiply(value, np.sin(dir_new))
        dy = dHy*h + np.multiply(value, np.cos(dir_new))

        # Candidate new FOA
        tryFOA_x = (pred_foa[0] + dx).round().squeeze()
        tryFOA_y = (pred_foa[1] + dy).round().squeeze()

        # Verifies if the candidate shift is located within the image
        validcord = np.nonzero(((tryFOA_x >= 0) & (tryFOA_x < nrow) &
                                (tryFOA_y >= 0) & (tryFOA_y < ncol)))[0]
        logger.verbose(f"Sampled {len(validcord)} valid candidate FOAs")

        if len(validcord) > 0:
            tryFOA_x = tryFOA_x[validcord].round()
            tryFOA_y = tryFOA_y[validcord].round()
            NcandFOAs = len(tryFOA_y)
            candx, candy = tryFOA_x, tryFOA_y

            if len(validcord) == 1:
                # Retains only the valid ones
                final_foa[0], final_foa[1] = candx, candy
            else:
                # Retains only the valid ones
                logger.verbose(f'Sampled {NcandFOAs} candidate new FOAS')

                # Computes the local visibility with respect to the FOA
                foaSize = round(max(nrow,ncol) / 6)
                visibility_radius = 1.5*foaSize

                # Choose the best FOA among the simulated
                # For each simulated FOA, computes how many preys /IPs
                # are within the visual search range. The IP which can get more preys survives.
                sampled_ip = np.concatenate((xCordIP[:,None], yCordIP[:,None]), axis=1)
                candidate_foas = np.concatenate((tryFOA_x[:,None], tryFOA_y[:,None]), axis=1)
                logger.verbose('Choosing the best one')

                # Performs a range search via kd-tree
                points_kdtree = cKDTree(sampled_ip)
                idxIP = points_kdtree.query_ball_point(candidate_foas, visibility_radius)

                # Get the best candidate
                maxnIP= -1
                for nCand in range(NcandFOAs):
                    temp = idxIP[nCand]
                    len_temp = len(temp)
                    if (len_temp > maxnIP):
                        maxnIP = len_temp
                        bestCandID = nCand

                final_foa = candidate_foas[bestCandID, :]

            final_foa = final_foa.round()
            accept = 1

        self.candx = candx
        self.candy = candy
        return final_foa, dir_new, accept

    def _create_circular_mask(self, center):
        """Creates a circular mask visualization of the FOA.

        Args:
            center (np.ndarray): (1, 2) vector representing the center of the FOA

        Returns:
            mask (np.ndarray): (h, w) binary mask
        """
        h = self.frame_sampling.n_rows
        w = self.frame_sampling.n_cols
        radius = self.foa_size

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        mask = np.ma.masked_where(mask == 1, mask)

        return mask
