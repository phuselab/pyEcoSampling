"""Static and space-time visual saliency detection by self-resemblance..

Seo's et al. present a novel unified framework for both static and space-time
saliency detection. Their method is a bottom-up approach and computes so-called
local regression kernels (i.e., local descriptors) from the given image (or a video),
which measure the likeness of a pixel (or voxel) to its surroundings. Visual saliency
is then computed using the said "self-resemblance" measure. The framework results in a
saliency map where each pixel (or voxel) indicates the statistical likelihood of saliency
of a feature matrix given its surrounding feature matrices. As a similarity measure,
matrix cosine similarity (a generalization of cosine similarity) is employed.
State of the art performance is demonstrated on commonly used human eye fixation data
(static scenes (N. Bruce & J. Tsotsos, 2006) and dynamic scenes (L. Itti & P. Baldi, 2006))
and some psychological patterns.

Note:
    Adapted from Matlab's version by Hae Jong on Apr 25, 2011 [1]_.

References
----------
.. [1] `Seo, H. J., & Milanfar, P. (2009). Static and space-time visual saliency detection
    by self-resemblance. Journal of vision, 9(12), 15-15.
    <https://jov.arvojournals.org/article.aspx?articleid=2122209>`_
"""

import numpy as np

class SelfRessemblance:
    """Static and space-time visual saliency detection by self-resemblance.

    Parameters:
        wsize (int): window size
        wsize_t (int): window size for time
        alpha (float): alpha parameter
        sigma (float): sigma parameter
        h (float): h parameter

    Attributes:
        wsize (int): window size
        wsize_t (int): window size for time
        alpha (float): alpha parameter
        sigma (float): sigma parameter
        h (float): h parameter
    """

    def __init__(self, wsize, wsize_t, alpha, sigma, h):
        self.wsize = wsize
        self.wsize_t = wsize_t
        self.sigma = sigma
        self.alpha = alpha
        self.h = h

    def three_D_LARK(self, fov_seq):
        """Compute 3-D LARK descriptors.

        Note:
            Adapted from Matlab's version by Hae Jong on Apr 25, 2011 [1]_.

        Args:
            fov_seq (np.ndarray): Foveated Map Sequence.

        Returns:
            lark (np.ndarray): 3D LARK descriptors
        """

        wsize = self.wsize
        h = self.h
        wsize_t = self.wsize_t
        alpha = self.alpha

        # Gradient calculation
        zx, zy, zt = np.gradient(fov_seq)
        M, N, T = fov_seq.shape

        win = (wsize-1) // 2
        win_t = (wsize_t-1) // 2


        win_width = np.array([win,win,win_t])


        zx = self._edge_mirror(zx,win_width)
        zy = self._edge_mirror(zy,win_width)
        zt = self._edge_mirror(zt,win_width)

        x1, x2, x3 = np.meshgrid(np.arange(-win, win+1), np.arange(-win, win+1), np.arange(-win_t, win_t+1))


        disk_filter = np.array([[0.0250785810238330, 0.145343947430219, 0.0250785810238330],
                                [0.145343947430219, 0.318309886183791, 0.145343947430219],
                                [0.0250785810238330, 0.145343947430219,	0.0250785810238330]])

        K = np.zeros((disk_filter.shape[0], disk_filter.shape[1], wsize_t))
        for k in range(0, wsize_t):
            K[:,:,k] = disk_filter
        K = np.transpose(K, (0,2,1))

        for k in range(0, wsize_t):
            K[:,:,k] = np.multiply(K[:,:,k], disk_filter)
        K = np.transpose(K, (2,1,0))

        for k in range(0, wsize_t):
            K[:,:,k] = np.multiply(K[:,:,k], disk_filter)
        K = np.transpose(K, (0,2,1))

        for k in range(0, wsize_t):
            K[:,:,k] = np.divide(K[:,:,k], K[win,win,k])

        len = np.sum(K)

        lambda_value = 1
        # Covariance matrices computation
        C11 = np.zeros((M, N, T))
        C12 = np.zeros((M, N, T))
        C22 = np.zeros((M, N, T))
        C13 = np.zeros((M, N, T))
        C23 = np.zeros((M, N, T))
        C33 = np.zeros((M, N, T))

        for i in range(0, M):
            for j in range(0, N):
                for k in range(0, T):

                    gx = np.multiply(zx[i:i+wsize, j:j+wsize, k:k+wsize_t], K)
                    gy = np.multiply(zy[i:i+wsize, j:j+wsize, k:k+wsize_t], K)
                    gt = np.multiply(zt[i:i+wsize, j:j+wsize, k:k+wsize_t], K)

                    G = np.concatenate((np.expand_dims(gx.flatten(), axis=1),
                                        np.expand_dims(gy.flatten(), axis=1),
                                        np.expand_dims(gt.flatten(), axis=1)), axis=1)

                    _, s, v, = np.linalg.svd(G)

                    S = np.zeros(3)
                    S[0] = (s[0] + lambda_value) / ( np.sqrt(s[1]*s[2]) + lambda_value)
                    S[1] = (s[1] + lambda_value) / ( np.sqrt(s[0]*s[2]) + lambda_value)
                    S[2] = (s[2] + lambda_value) / ( np.sqrt(s[1]*s[0]) + lambda_value)


                    tmp = (S[0] * np.expand_dims(v[:,0], axis=1) @ np.expand_dims(v[:,0], axis=1).T +
                        S[1] * np.expand_dims(v[:,1], axis=1) @ np.expand_dims(v[:,1], axis=1).T +
                        S[2] * np.expand_dims(v[:,2], axis=1) @ np.expand_dims(v[:,2], axis=1).T) * \
                        ((s[0] * s[1] * s[2] + 0.0000001) / len)**alpha


                    C11[i,j,k] = tmp[0,0]
                    C12[i,j,k] = tmp[0,1]
                    C22[i,j,k] = tmp[1,1]
                    C13[i,j,k] = tmp[0,2]
                    C23[i,j,k] = tmp[1,2]
                    C33[i,j,k] = tmp[2,2]

        C11 = self._edge_mirror(C11, win_width)
        C12 = self._edge_mirror(C12, win_width)
        C22 = self._edge_mirror(C22, win_width)
        C23 = self._edge_mirror(C23, win_width)
        C33 = self._edge_mirror(C33, win_width)
        C13 = self._edge_mirror(C13, win_width)

        x13 = 2*np.multiply(x1, x3)
        x12 = 2*np.multiply(x1, x2)
        x23 = 2*np.multiply(x2, x3)
        x11 = np.square(x1)
        x22 = np.square(x2)
        x33 = np.square(x3)

        new_shape = [1, (wsize**2)*wsize_t]
        new_final_shape = [M, N, T, wsize, wsize, wsize_t]

        x1x1 = np.reshape(np.tile(np.reshape(x11, new_shape), (M*N*T,1)), new_final_shape)
        x2x2 = np.reshape(np.tile(np.reshape(x22, new_shape), (M*N*T,1)), new_final_shape)
        x3x3 = np.reshape(np.tile(np.reshape(x33, new_shape), (M*N*T,1)), new_final_shape)
        x1x2 = np.reshape(np.tile(np.reshape(x12, new_shape), (M*N*T,1)), new_final_shape)
        x1x3 = np.reshape(np.tile(np.reshape(x13, new_shape), (M*N*T,1)), new_final_shape)
        x2x3 = np.reshape(np.tile(np.reshape(x23, new_shape), (M*N*T,1)), new_final_shape)

        # % Geodesic distance computation between a center and surrounding voxels
        lark = np.zeros((M, N, T, wsize, wsize, wsize_t))

        for i in range(0, wsize):
            for j in range(0, wsize):
                for k in range(0, wsize_t):
                    temp = np.multiply(C11[i:i+M,j:j+N,k:k+T], x1x1[:,:,:,i,j,k]) + \
                        np.multiply(C22[i:i+M,j:j+N,k:k+T], x2x2[:,:,:,i,j,k]) + \
                        np.multiply(C33[i:i+M,j:j+N,k:k+T], x3x3[:,:,:,i,j,k]) + \
                        np.multiply(C12[i:i+M,j:j+N,k:k+T], x1x2[:,:,:,i,j,k]) + \
                        np.multiply(C13[i:i+M,j:j+N,k:k+T], x1x3[:,:,:,i,j,k]) + \
                        np.multiply(C23[i:i+M,j:j+N,k:k+T], x2x3[:,:,:,i,j,k])
                    lark[:,:,:,i,j,k] = temp


        # Convert geodesic distance to self-similarity
        lark = np.exp(-lark*0.5/h**2)
        lark = np.reshape(lark, [M, N, T, (wsize**2)*wsize_t])

        return lark

    def space_time_saliency_map(self, lark):
        """Compute Space-time Self-Resemblance.

        Note:
            Adapted from Matlab's version by Hae Jong on Apr 25, 2011 [1]_.

        Args:
            lark (np.array): 3D LARK descriptors

        Returns:
            Space-time Saliency Map
        """
        wsize = self.wsize
        size_t = self.wsize_t
        sigma = self.sigma

        win = (wsize-1) // 2
        win_t = (size_t-1) // 2

        width = np.array([win,win,win_t])

        ls0, ls1, ls2, ls3 = lark.shape

        # To avoid edge effect, we use mirror padding.
        lark1 = np.zeros((ls0+2*win, ls1+2*win, ls2+2*win_t, ls3))
        for i in range(0, lark.shape[3]):
            lark1[:,:,:,i] = self._edge_mirror(lark[:,:,:,i], width)

        # Precompute Norm of center matrices and surrounding matrices
        norm_C = np.zeros((ls0, ls1, ls2))
        for i in range(0, ls0):
            for j in range(0, ls1):
                for l in range(0, ls2):
                    norm_C[i,j,l] = np.linalg.norm(np.squeeze(lark[i,j,l,:]))

        norm_S = np.zeros((ls0+2*win, ls1+2*win, ls2+2*win_t))
        norm_S[:,:,:] = self._edge_mirror(norm_C, width)

        new_shape = [ls0*ls1*ls2, ls3]
        center = np.reshape(lark, new_shape)
        new_shape[1] = 1
        norm_C = np.reshape(norm_C, new_shape)

        saliency_map = np.zeros(norm_C.shape)

        for i in range(0, wsize):
            for j in range(0, wsize):
                for l in range(0, size_t):
                    new_shape[1] = ls3
                    # LARK1(i:i+size(LARK,1)-1,j:j+size(LARK,2)-1,l:l+size(LARK,3)-1,:)
                    lark_reshaped = np.reshape(lark1[i:i+ls0,j:j+ls1,l:l+ls2,:], new_shape)
                    temp = np.sum(np.multiply(center, lark_reshaped), axis=1)
                    # Compute inner product between a center and surrounding matrices
                    new_shape[1] = 1
                    norm_s_reshaped = np.reshape(norm_S[i:i+ls0, j:j+ls1, l:l+ls2], new_shape)
                    temp = np.divide(np.expand_dims(temp, axis=1), np.multiply(norm_C, norm_s_reshaped))
                    # compute self-resemblance using matrix cosine similarity
                    saliency_map = np.add(saliency_map, np.exp(np.divide((-1+temp), np.square(sigma))))

        # Final saliency map values
        saliency_map = np.divide(1, saliency_map)

        new_shape = [ls0, ls1, ls2]
        saliency_map = np.reshape(saliency_map, new_shape)
        return saliency_map



    def _edge_mirror(self, x, width):
        """Pad with mirroring the edges of the image.

        Pads with the reflection of the vector mirrored
        on the first and last values of the vector along each axis.

        Note:
            Adapted from Matlab's version by Hae Jong on Apr 25, 2011.

        Args:
            x (np.ndarray): Image to be reflected.
            width (vector): Width for each axis of the padding.

        Returns:
            z: Mirror padded image.

        """
        width = width.astype(int)
        end = -1
        y = np.concatenate((x[:, 2:width[1]:-1,:], x, x[: ,end:end-width[1]:-1,:]), axis=1)
        y = np.concatenate((y[2:width[0]:-1, :,:], y, y[end:end-width[0]:-1, :,:]), axis=0)
        z = np.concatenate((y[:, :, 2:width[2]:-1], y, y[:, :, end:end-width[2]:-1]), axis=2)
        return z


