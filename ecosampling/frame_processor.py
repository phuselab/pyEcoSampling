import os
import numpy as np

from utils.filters import mk_gaussian
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize
from config import GeneralConfig, SaliencyConfig

from utils.logger import Logger

logger = Logger(__name__)

class FrameProcessor:

    def __init__(self):
        # Number of Frames to process in a step
        self.n_frames = SaliencyConfig.WSIZE_T
        self.frame_offset = GeneralConfig.OFFSET
        # Image List Setup
        self.img_list = [os.path.join(GeneralConfig.FRAME_DIR, image) for image in os.listdir(GeneralConfig.FRAME_DIR) if image.endswith(".jpg")]
        self.img_list.sort()
        # Total Frames in list
        self.total_frames = len(self.img_list)

        r, c = io.imread(self.img_list[0])[:,:,0].shape
        self.n_rows = r
        self.n_cols = c

        I_shape = (self.n_rows, self.n_cols, self.n_frames)
        self.I = np.zeros(I_shape)
        self.foveated_I = np.zeros(I_shape)
        self.current_frame = np.zeros((self.n_rows, self.n_cols))
        self.current_foveated_frame = np.zeros((self.n_rows, self.n_cols))

    def frame_resize_orginal(self, frame):
        return resize(frame, (self.n_rows, self.n_cols), order=1)


    def read_frames(self, frame_idx):
        logger.verbose("Data acquisition")
        # Reading three consecutive frames
        # Get previous frame
        img_path = self.img_list[frame_idx-self.frame_offset]
        pred_frame = io.imread(img_path)
        # Get current frame
        img_path = self.img_list[frame_idx]
        curr_frame = io.imread(img_path)
        # Get subsequent frame
        img_path = self.img_list[frame_idx+self.frame_offset]
        next_frame = io.imread(img_path)

        # Converting to grey level
        self.I[:,:,0] = rgb2gray(pred_frame)
        self.I[:,:,1] = rgb2gray(curr_frame)
        self.I[:,:,2] = rgb2gray(next_frame)
        self.current_frame = curr_frame

        return self.I

    def foveated_imaging(self, final_foa, I):
        logger.verbose("Makes a foa dependent image")
        #   Using:
        #   IM = mkGaussian(SIZE, COVARIANCE, MEAN, AMPLITUDE)

        #   Compute a matrix with dimensions SIZE (a [Y X] 2-vector, or a
        #   scalar) containing a Gaussian function, centered at pixel position
        #   specified by MEAN (default = (size+1)/2), with given COVARIANCE (can
        #   be a scalar, 2-vector, or 2x2 matrix.  Default = (min(size)/6)^2),
        #   and AMPLITUDE.  AMPLITUDE='norm' (default) will produce a
        #   probability-normalized function.  All but the first argument are
        #   optional.
        #   from Eero Simoncelli, 6/96.
        size = np.array([self.n_rows, self.n_cols])
        cov = (np.min(size)/1.5)**2.5
        conjugate_T_foa = final_foa.conj().T

        foa_filter = mk_gaussian(size, cov, conjugate_T_foa, 1)

        self.foveated_I[:,:,0] = np.multiply(I[:,:,0].astype('double'),
                                            foa_filter)
        self.foveated_I[:,:,1] = np.multiply(I[:,:,1].astype('double'),
                                             foa_filter)
        self.foveated_I[:,:,2] = np.multiply(I[:,:,2].astype('double'),
                                             foa_filter)

        self.current_foveated_frame = self.foveated_I[:,:,1]

        return self.foveated_I

    def reduce_frames(self, foveated_I, reduced_rows=64, reduced_cols=64):
        logger.verbose("Reducing frames size for feature processing")
        # Reducing the frame to [64 64] dimension suitable for feature extraction
        reduced_frame = np.zeros((reduced_rows, reduced_cols, self.n_frames))

        # Bilinear by default
        S = resize(foveated_I[:,:,0].astype('double'), (64, 64), order=1)
        reduced_frame [:,:,0] = np.divide(S, np.std(S[:]))

        S = resize(foveated_I[:,:,1].astype('double'), (64, 64), order=1)
        reduced_frame [:,:,1] = S / np.std(S[:])

        S = resize(foveated_I[:,:,2].astype('double'), (64, 64), order=1)
        reduced_frame [:,:,2] = S / np.std(S[:])

        reduced_frame  = reduced_frame  / np.std(reduced_frame [:])

        return reduced_frame
