import numpy as np
import skimage.morphology

def EdgeMirror3(x, width):
    width = width.astype(int)
    end = -1
    y = np.concatenate((x[:, 2:width[1]:-1,:], x, x[: ,end:end-width[1]:-1,:]), axis=1)
    y = np.concatenate((y[2:width[0]:-1, :,:], y, y[end:end-width[0]:-1, :,:]), axis=0)
    z = np.concatenate((y[:, :, 2:width[2]:-1], y, y[:, :, end:end-width[2]:-1]), axis=2)
    return z

def ThreeDLARK(fov_seq, params):

    wsize = params["wsize"]
    h = params["h"]
    wsize_t = params["wsize_t"]
    alpha = params["alpha"]

    # Gradient calculation
    zx, zy, zt = np.gradient(fov_seq)
    M, N, T = fov_seq.shape

    win = (wsize-1) // 2
    win_t = (wsize_t-1) // 2


    win_width = np.array([win,win,win_t])


    zx = EdgeMirror3(zx,win_width)
    zy = EdgeMirror3(zy,win_width)
    zt = EdgeMirror3(zt,win_width)

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

    C11 = EdgeMirror3(C11, win_width)
    C12 = EdgeMirror3(C12, win_width)
    C22 = EdgeMirror3(C22, win_width)
    C23 = EdgeMirror3(C23, win_width)
    C33 = EdgeMirror3(C33, win_width)
    C13 = EdgeMirror3(C13, win_width)

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
