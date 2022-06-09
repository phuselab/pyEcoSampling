"""File for plotting data with matplotlib.

Authors:
    Giuseppe Boccignone <giuseppe.boccignone@unimi.it>
    Renato Nobre <renato.avellarnobre@studenti.unimi.it>

Changes:
    12/12/2012  First Edition Matlab
    31/05/2022  Python Edition
"""

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse, Circle

from config import GeneralConfig

class Plotter:
    """Class for plotting data with matplotlib.

    Note:
        Don't need to instantiate this class, unless
        you want to plot the entire visualization with
        ``plot_visualization``.

    Args:
        num_line (int, optional): Number of lines to plot.
            Defaults to 2.
        num_pics_line (int, optional): Number of pictures to plot in each line.
            Defaults to 5.
        fig_size (tuple, optional): Size of the figure.
            Defaults to (15, 5).

    Attributes:
        fig (plt.figure.Figure): Figure to plot the visualization.
        axes (plt.axes.Axes): Axes to plot the visualization.
    """

    def __init__(self, num_line=2, num_pics_line=5, fig_size=(15, 5)):
        fig, ax = plt.subplots(num_line, num_pics_line, figsize=fig_size)
        self.fig = fig
        self.axes = ax
        plt.ion()
        plt.show()

    def _clean_axes(self):
        """Clean axes of the plot visualization."""
        for i in range(2):
            for j in range(5):
                self.axes[i, j].clear()

    @classmethod
    def _configure_axis(cls, ax, title, keep_axis=False):
        """Configure axis interface.

        Wrap common aspects of plot axis.

        Args:
            ax (axes.Axes): Axes to plot the image
            title (str): Title of the plot
            keep_axis (bool, optional): Whether to show or hide the axis.
                Defaults to False.
        """
        if not keep_axis:
            ax.set_axis_off()
        ax.set_title(title)

    @classmethod
    def _image_plot(cls, ax, image, title, cmap='viridis', interpol=None):
        """Image plot interface.

        Wrap common aspects of image plotting.

        Args:
            ax (axes.Axes): Axes to plot the image
            image (np.ndarray): Image to plot.
            title (str): Title of the plot
            cmap (str, optional): Color map. Defaults to 'viridis'.
            interpol (str, optional): Interpolation type. Defaults to None.
        """
        cls._configure_axis(ax, title)
        ax_image = ax.imshow(image, cmap=cmap, interpolation=interpol)
        return ax_image

    def save_foa_values(self, foa_values, n_obs):
        """Save foa numpy values.

        Args:
            foa_values (np.ndarray): Foa values to save.
        """

        # Save foa numpy values
        if GeneralConfig.SAVE_FOA_ONFILE:
            path = self._create_result_folder('', n_obs)
            with open(path + '/foa_values.npy', 'wb') as f:
                np.save(f, foa_values)

    def save_complexity(self, complexity, n_obs):
        """Save complexy data plots an numpy values.

        Args:
            data (dict): Data dictionary
        """
        if GeneralConfig.SAVE_COMPLEXITY_ONFILE:
            path = self._create_result_folder('', n_obs)

            # Plot Final Order/Disorder
            fig, ax = plt.subplots(figsize=(12,6), dpi=100)
            self.plot_order_disorder(ax, complexity)
            fig.savefig(path + '/order_disorder_plot.png')
            plt.close(fig)

            # Plot final complexity
            fig, ax = plt.subplots(figsize=(12,6), dpi=100)
            self.plot_complexity(ax, complexity)
            fig.savefig(path + '/complexity_plot.png')
            plt.close(fig)

            # Save numpy values
            with open(path + '/order.npy', 'wb') as f:
                np.save(f, complexity.order)
            with open(path + '/disorder.npy', 'wb') as f:
                np.save(f, complexity.disorder)
            with open(path + '/complexity.npy', 'wb') as f:
                np.save(f, complexity.complexity)

    def plot_visualization(self, data, frame_num, pause_time=0.001):
        """Plot complete visualization for instance.

        Plot all 10 plots for every frame.

        Args:
            data (dict): Data dictionary
            pause_time (float, optional): Time to visualize the plot.
                Defaults to 0.001 (for GPU issues).
        """
        self._clean_axes()
        self.plot_original_frame(self.axes[0, 0], data["frame_sampling"])
        self.plot_foveated_frame(self.axes[0, 1], data["frame_sampling"])
        self.plot_feature_map(self.axes[0, 2], data["feature_map"])
        self.plot_saliency_map(self.axes[0, 3], data["saliency_map"])
        self.plot_proto_objects(self.axes[0, 4], data["frame_sampling"],
                                data["proto_params"], data["num_proto"])
        self.plot_interest_points(self.axes[1, 0], data["frame_sampling"],
                                  data["circle_coords"], data["gaze_sampler"])
        self.plot_empirical_dists(self.axes[1, 1], data["hist_mat"])
        self.plot_order_disorder(self.axes[1, 2], data["complexity"])
        self.plot_complexity(self.axes[1, 3], data["complexity"])
        self.plot_sampled_FOA(self.axes[1, 4], data["frame_sampling"],
                              data["gaze_sampler"])

        plt.tight_layout()
        plt.pause(pause_time)


    def plot_original_frame(cls, axes, frame_sampling):
        """Plot original frame.

        Args:
            axes (axes.Axes): Axes to plot the image
            frame_sampling (obj): Frame sampling object.
        """
        cls._image_plot(axes, frame_sampling.show_frame, 'Current frame')

    def plot_foveated_frame(cls, axes, frame_sampling):
        """Plot foveated frame.

        Args:
            axes (axes.Axes): Axes to plot the image
            frame_sampling (obj): Frame sampling object.
        """
        cls._image_plot(axes, frame_sampling.show_foveated_frame, 'Foveated frame', cmap='gray')

    def plot_feature_map(cls, axes, feature_map):
        """Plot feature map.

        Args:
            axes (axes.Axes): Axes to plot the image
            feature_map (np.ndarray): Feature map frame to plot.
        """
        cls._image_plot(axes, feature_map, 'Feature map', cmap='gray')

    def plot_saliency_map(cls, axes, saliency_map):
        """Plot saliency map.

        Args:
            axes (axes.Axes): Axes to plot the image
            saliency_map (np.ndarray): Salience map frame to plot.
        """
        cls._image_plot(axes, saliency_map, 'Salience map', cmap='jet')

    def plot_proto_objects(cls, axes, frame_sampler, proto_params, num_proto):
        """Plot Proto Objects.

        Args:
            axes (axes.Axes): Axes to plot the image
            frame_sampler (obj): Frame sampling object.
            proto_params (obj): protoparams objetcs to plot.
            num_proto (int): Number of protoparameres.
        """

        if num_proto > 0:
            cls._image_plot(axes, frame_sampler.show_frame, 'Proto-Objects')
            cls._image_plot(axes, proto_params.show_proto, 'Proto-Objects',
                            cmap='gray', interpol='nearest')

            for p in range(proto_params.nV):
                ((centx,centy), (width,height), angle) = proto_params.a[p]
                elli = Ellipse((centx,centy), width, height, angle)
                elli.set_ec('yellow')
                elli.set_fill(False)
                axes.add_artist(elli)


    def plot_interest_points(cls, axes, frame_sampler, circle_coords, gaze_sampler):
        """Plot Interest points.

        Args:
            axes (axes.Axes): Axes to plot the image
            frame_sampler (obj): Frame sampling object.
            circle_coords (np.ndarray): Coordinates of interest points
            gaze_sampler (obj): gaze_sampler object
        """
        cls._image_plot(axes, frame_sampler.show_frame, "Sampled Interest Points (IP)")
        # Show image with region marked
        xCoord, yCoord = circle_coords
        candx, candy = gaze_sampler.candx, gaze_sampler.candy
        for b in range(yCoord.shape[0]):
            circle = Circle((xCoord[b],yCoord[b]), 4, color='r', lw=1)
            axes.add_artist(circle)

        for idc in range(len(candx)):
            circle = Circle((candx[idc], candy[idc]), 4, color='y', lw=2)
            axes.add_artist(circle)

        circle = Circle((gaze_sampler.candidate_foa[0], gaze_sampler.candidate_foa[1]), 10, color='g', lw=6)
        axes.add_artist(circle)

    def plot_empirical_dists(cls, axes, hist_mat):
        """Plot IP Empirical distribution.

        Args:
            axes (axes.Axes): Axes to plot the image
            hist_mat (np.ndarray): 2D histogram array, already transposed to original
        """
        cls._configure_axis(axes, "IP Empirical Distribution")
        sns.heatmap(hist_mat, linewidth=0.2, cbar=False, cmap='jet', ax=axes)


    def plot_order_disorder(cls, axes, complexity, lw=2):
        """Plot order/disorder.

        Args:
            axes (axes.Axes): Axes to plot the image
            complexity (obj): Complexity object.
            lw (int, optional): Line width. Defaults to 2.
        """
        cls._configure_axis(axes, "Order/Disorder", keep_axis=True)
        axes.plot(complexity.disorder, 'r--', label='Disorder', linewidth=lw)
        axes.plot(complexity.order, 'g-', label='Order', linewidth=lw)
        axes.legend(loc='upper right')


    def plot_complexity(cls, axes, complexity, lw=2):
        """Plot complexity curves.

        Args:
            axes (axes.Axes): Axes to plot the image
            complexity (obj): Complexity object.
            lw (int, optional): Line width. Defaults to 2.
        """
        cls._configure_axis(axes, "Complexity" , keep_axis=True)
        axes.plot(complexity.complexity, label='Complexity', linewidth=lw)

    def plot_sampled_FOA(cls, axes, frame_sampler, gaze_sampler):
        """Plot final FOA.

        Args:
            axes (axes.Axes): Axes to plot the image
            frame_sampler (obj): Frame sampling object.
            gaze_sampler (obj): Gaze sampler object.
        """
        cls._image_plot(axes, frame_sampler.show_frame, "Final FOA")
        cls._image_plot(axes, gaze_sampler.show, "Final FOA", cmap='gray', interpol='nearest')


    def _create_result_folder(cls, plot_type_folder, n_obs):
        """Create result folder.

        Args:
            plot_type_folder (str): Plot type folder.

        Returns:
            str: Path to the result folder.
        """

        results_folder = GeneralConfig.RESULTS_DIR

        if GeneralConfig.TOTAL_OBSERVERS > 1:
            results_folder = results_folder + f"/obs_{n_obs+1}/"

        # Create result folder
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        # Create subfolders
        if not os.path.exists(results_folder + plot_type_folder):
            os.makedirs(results_folder + plot_type_folder)

        return results_folder + plot_type_folder


    def _configure_axis_save(cls, fig, ax, save_name, frame_num, n_obs):
        """Configure axis to save a clean image.

        Args:
            fig (obj): Matplotlib figure
            ax (axes.Axes): Axes to plot the image
            save_name (string): Name of the image to save and path to create.
            frame_num (int): Number of the frame
        """
        path = cls._create_result_folder(f'{save_name}', n_obs)
        plt.axis('off')
        ax.set_title('')
        fig.savefig(path + f'/{save_name}_{frame_num}.png', bbox_inches='tight', pad_inches = 0)
        ax.clear()

    def save_imgs(self, data, frame_num, n_obs):
        """"Save images to disk.

        Args:
            data (dict): Data dictionary
            frame_num (int): Number of the frame
        """

        fig, ax = plt.subplots(figsize=(12,6), dpi=100)

        if GeneralConfig.SAVE_FOV_IMG:
            self.plot_foveated_frame(ax, data["frame_sampling"])
            self._configure_axis_save(fig, ax, 'foveated', frame_num, n_obs)

        if GeneralConfig.SAVE_SAL_IMG:
            self.plot_saliency_map(ax, data["saliency_map"])
            self._configure_axis_save(fig, ax, 'salience', frame_num, n_obs)

        if GeneralConfig.SAVE_PROTO_IMG:
            self.plot_proto_objects(ax, data["frame_sampling"],
                                    data["proto_params"], data["num_proto"])
            self._configure_axis_save(fig, ax, 'protos', frame_num, n_obs)

        if GeneralConfig.SAVE_IP_IMG:
            self.plot_interest_points(ax, data["frame_sampling"],
                                      data["circle_coords"], data["gaze_sampler"])
            self._configure_axis_save(fig, ax, 'ips', frame_num, n_obs)

        if GeneralConfig.SAVE_HISTO_IMG:
            self.plot_empirical_dists(ax, data["hist_mat"])
            self._configure_axis_save(fig, ax, 'empirical_dists', frame_num, n_obs)

        if GeneralConfig.SAVE_FOA_IMG:
            self.plot_sampled_FOA(ax, data["frame_sampling"], data["gaze_sampler"])
            self._configure_axis_save(fig, ax, 'foa', frame_num, n_obs)

        plt.close(fig)

