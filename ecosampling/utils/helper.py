import numpy as np

def EdgeMirror3(x, width):
    width = width.astype(int)
    end = -1
    y = np.concatenate((x[:, 2:width[1]:-1,:], x, x[: ,end:end-width[1]:-1,:]), axis=1)
    y = np.concatenate((y[2:width[0]:-1, :,:], y, y[end:end-width[0]:-1, :,:]), axis=0)
    z = np.concatenate((y[:, :, 2:width[2]:-1], y, y[:, :, end:end-width[2]:-1]), axis=2)
    return z
