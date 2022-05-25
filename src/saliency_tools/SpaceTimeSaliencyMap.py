
# % Compute Space-time Self-Resemblance

# % [RETURNS]
# % SM   : Space-time Saliency Map
# %
# % [PARAMETERS]
# % img   : Input image
# % LARK  : A collection of LARK descriptors
# % param : parameters

# % [HISTORY]

# % Apr 25, 2011 : created by Hae Jong

from turtle import width
import numpy as np

def EdgeMirror3(x, width):
    width = width.astype(int)
    end = -1
    y = np.concatenate((x[:, 2:width[1]:-1,:], x, x[: ,end:end-width[1]:-1,:]), axis=1)
    y = np.concatenate((y[2:width[0]:-1, :,:], y, y[end:end-width[0]:-1, :,:]), axis=0)
    z = np.concatenate((y[:, :, 2:width[2]:-1], y, y[:, :, end:end-width[2]:-1]), axis=2)
    return z


def SpaceTimeSaliencyMap(seq, lark, s_param):
    wsize = s_param["wsize"]
    size_t = s_param["wsize_t"]
    sigma = s_param["sigma"]

    win = (wsize-1) // 2
    win_t = (size_t-1) // 2

    width = np.array([win,win,win_t])

    ls0, ls1, ls2, ls3 = lark.shape

    # To avoid edge effect, we use mirror padding.
    lark1 = np.zeros((ls0+2*win, ls1+2*win, ls2+2*win_t, ls3))
    for i in range(0, lark.shape[3]):
        lark1[:,:,:,i] = EdgeMirror3(lark[:,:,:,i], width)

    # Precompute Norm of center matrices and surrounding matrices
    norm_C = np.zeros((ls0, ls1, ls2))
    for i in range(0, ls0):
        for j in range(0, ls1):
            for l in range(0, ls2):
                norm_C[i,j,l] = np.linalg.norm(np.squeeze(lark[i,j,l,:]))

    norm_S = np.zeros((ls0+2*win, ls1+2*win, ls2+2*win_t))
    norm_S[:,:,:] = EdgeMirror3(norm_C, width)

    # M,N,T = seq.shape

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
                # norm_S(i:i+size(LARK,1)-1,j:j+size(LARK,2)-1,l:l+size(LARK,3)-1)
                norm_s_reshaped = np.reshape(norm_S[i:i+ls0, j:j+ls1, l:l+ls2], new_shape)

                temp = np.divide(np.expand_dims(temp, axis=1), np.multiply(norm_C, norm_s_reshaped))

                # compute self-resemblance using matrix cosine similarity
                saliency_map = np.add(saliency_map, np.exp(np.divide((-1+temp), np.square(sigma))))

    # Final saliency map values
    saliency_map = np.divide(1, saliency_map)

    new_shape = [ls0, ls1, ls2]
    saliency_map = np.reshape(saliency_map, new_shape)
    return saliency_map

