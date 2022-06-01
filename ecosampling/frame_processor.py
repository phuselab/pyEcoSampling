"""Frame Processor class file.

Authors:
    - Giuseppe Boccignone <giuseppe.boccignone@unimi.it>
    - Renato Nobre <renato.avellarnobre@studenti.unimi.it>

Changes:
    - 12/12/2012  First Edition Matlab
    - 31/05/2022  Python Edition
"""


import os

import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize

from config import GeneralConfig, SaliencyConfig
from utils.helper import mk_gaussian
from utils.logger import Logger

logger = Logger(__name__)

class FrameProcessor:
    """Set of functions to handle image input and create frames.

    This class has functions to create the window size frame from images
    in a local dir, apply foveated filtering to a frame, reduce the frame
    size and increase image size.

    Attributes:
        n_frames (int): Number of frames to process in a step.
        frame_offset (int): Offset between frames.
        img_list (list): List of images in the local dir.
        total_frames (int): Total number of images in the local dir.
        n_rows (int): Number of rows in the image.
        n_cols (int): Number of columns in the image.
        I (np.ndarray): 3 time-step frame.
        foveated_I (np.ndarray): 3 time-step foveated frame.
        show_frame (np.ndarray): Current frame for visualization.
        show_foveated_frame (np.ndarray): Current foveated frame for visualization.
    """

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
        self.show_frame = np.zeros((self.n_rows, self.n_cols))
        self.show_foveated_frame = np.zeros((self.n_rows, self.n_cols))

    def frame_resize_orginal(self, frame):
        """Resize frame to original size

        Use bilinear interpolation

        Args:
            frame (np.ndarray): Small size frame

        Returns:
            np.ndarray: Original size frame
        """
        return resize(frame, (self.n_rows, self.n_cols), order=1)


    def read_frames(self, frame_idx):
        """Read current, next and previous image, creating a frame.

        Reading include opening the files, converting them
        to grayscale, and creating a 3 time-step frame.

        Note:
            Also save the frames in the object instance.

        Args:
            frame_idx (int): Current frame indices

        Returns:
            I: Grayscale image frame
        """

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
        self.show_frame = curr_frame

        return self.I

    def foveated_imaging(self, foa, I):
        """Apply foveated imaging to the Grayscale frame.

        To convert to the foveated version this function apply a gaussian
        filter in the field of attention (``foa``) of the image.

        Args:
            foa (np.ndarray): Field of attention.
            I (np.ndarray): Grayscale frame

        Note:
            Also save the frames in the object instance.

        Returns:
            Foveated frame
        """
        logger.verbose("Makes a foa dependent image")

        size = np.array([self.n_rows, self.n_cols])
        cov = (np.min(size)/1.5)**2.5
        conjugate_T_foa = foa.conj().T

        foa_filter = mk_gaussian(size, cov, conjugate_T_foa, 1)

        self.foveated_I[:,:,0] = np.multiply(I[:,:,0].astype('double'),
                                            foa_filter)
        self.foveated_I[:,:,1] = np.multiply(I[:,:,1].astype('double'),
                                             foa_filter)
        self.foveated_I[:,:,2] = np.multiply(I[:,:,2].astype('double'),
                                             foa_filter)

        self.show_foveated_frame = self.foveated_I[:,:,1]

        return self.foveated_I

    def reduce_frames(self, foveated_I, reduced_rows=64, reduced_cols=64):
        """Reduce the frame to a desired size.

        The main purpose of this reduction is to compute the features.

        Args:
            foveated_I (np.ndarray): Frame to be reduced.
            reduced_rows (int, optional): Final desired row size. Defaults to 64.
            reduced_cols (int, optional): Final desired col size. Defaults to 64.

        Returns:
            The reduced frame.
        """
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
