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

    def _save_imgs(self, data, frame_num):

        fig, ax = plt.subplots()

        if GeneralConfig.SAVE_FOV_IMG:
            path = self.create_result_folder('results', 'foveated')
            self.plot_foveated_frame(ax, data["foveated_frame"], title=f'Foveated Frame #{frame_num}')
            fig.savefig(path + f'/Foveated_{frame_num}.png')
            ax.clear()

        if GeneralConfig.SAVE_SAL_IMG:
            path = self.create_result_folder('results', 'salience')
            self.plot_saliency_map(ax, data["saliency_map"], title=f'Salience #{frame_num}')
            fig.savefig(path + f'/Salience_{frame_num}.png')
            ax.clear()

        if GeneralConfig.SAVE_PROTO_IMG:
            path = self.create_result_folder('results', 'proto_objects')
            self.plot_proto_objects(ax, data["original_frame"],
                                    data["proto_params"], data["num_proto"],
                                    title=f"Proto-Objects #{frame_num}")
            fig.savefig(path + f'/Proto_{frame_num}.png')
            ax.clear()

        if GeneralConfig.SAVE_IP_IMG:
            path = self.create_result_folder('results', 'interest_points')
            self.plot_interest_points(ax, data["original_frame"], data["circle_coords"],
                                      data["gaze_sampler"],
                                      title=f"Sampled Interest Points (IP) #{frame_num}")
            fig.savefig(path + f'/IP_{frame_num}.png')
            ax.clear()

        if GeneralConfig.SAVE_HISTO_IMG:
            path = self.create_result_folder('results', 'emprical_distribution')
            self.plot_empirical_dists(ax, data["hist_mat"], title=f"IP Empirical Distribution #{frame_num}")
            fig.savefig(path + f'/Histo_{frame_num}.png')
            ax.clear()

        plt.close(fig)

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


    def plot_visualization(self, data, frame_num, pause_time=0.001):
        """Plot complete visualization for instance.

        Plot all 10 plots for every frame.

        Args:
            data (dict): _description_
            pause_time (float, optional): Time to visualize the plot.
                Defaults to 0.001 (for GPU issues).
        """
        self._clean_axes()
        self.plot_original_frame(self.axes[0, 0], data["original_frame"])
        self.plot_foveated_frame(self.axes[0, 1], data["foveated_frame"])
        self.plot_feature_map(self.axes[0, 2], data["feature_map"])
        self.plot_saliency_map(self.axes[0, 3], data["saliency_map"])
        self.plot_proto_objects(self.axes[0, 4], data["original_frame"],
                                data["proto_params"], data["num_proto"])
        self.plot_interest_points(self.axes[1, 0], data["original_frame"],
                                  data["circle_coords"], data["gaze_sampler"])
        self.plot_empirical_dists(self.axes[1, 1], data["hist_mat"])
        self.plot_order_disorder(self.axes[1, 2], data["order"], data["disorder"])
        self.plot_complexity(self.axes[1, 3], data["complexity"])
        self.plot_sampled_FOA(self.axes[1, 4], data["original_frame"],
                              data["foa_center"], data["foa_radius"], title="Final FOA")

        plt.tight_layout()
        self._save_imgs(data, frame_num)
        plt.pause(pause_time)


    def plot_original_frame(cls, axes, original_frame, title='Current frame'):
        """Plot original frame.

        Args:
            axes (axes.Axes): Axes to plot the image
            original_frame (np.ndarray): Original frame to plot.
            title (str, optional): Title of the plot. Defaults to 'Current frame'.
        """
        cls._image_plot(axes, original_frame, title)

    def plot_foveated_frame(cls, axes, foveated_frame, title='Foveated frame'):
        """Plot foveated frame.

        Args:
            axes (axes.Axes): Axes to plot the image
            foveated_frame (np.ndarray): Fovelated frame to plot.
            title (str, optional): Title of the plot. Defaults to 'Foveated frame'.
        """
        cls._image_plot(axes, foveated_frame, title, cmap='gray')

    def plot_feature_map(cls, axes, feature_map, title='Feature map'):
        """Plot feature map.

        Args:
            axes (axes.Axes): Axes to plot the image
            feature_map (np.ndarray): Feature map frame to plot.
            title (str, optional): Title of the plot. Defaults to 'Feature map'.
        """
        cls._image_plot(axes, feature_map, title, cmap='gray')

    def plot_saliency_map(cls, axes, saliency_map, title='Salience map'):
        """Plot saliency map.

        Args:
            axes (axes.Axes): Axes to plot the image
            saliency_map (np.ndarray): Salience map frame to plot.
            title (str, optional): Title of the plot. Defaults to 'Salience map'.
        """
        cls._image_plot(axes, saliency_map, title, cmap='jet')

    def plot_proto_objects(cls, axes, current_frame, proto_params, num_proto, title='Proto-Objects'):
        """Plot Proto Objects.

        Args:
            axes (axes.Axes): Axes to plot the image
            current_frame (np.ndarray): Original frame to plot.
            proto_params (obj): protoparams objetcs to plot.
            num_proto (int): Number of protoparameres.
            title (str, optional): Title of the plot. Defaults to 'Proto-Objects'.
        """

        if num_proto > 0:
            cls._image_plot(axes, current_frame, title)
            cls._image_plot(axes, proto_params.show_proto, title,
                            cmap='gray', interpol='nearest')

            for p in range(proto_params.nV):
                ((centx,centy), (width,height), angle) = proto_params.a[p]
                elli = Ellipse((centx,centy), width, height, angle)
                elli.set_ec('yellow')
                elli.set_fill(False)
                axes.add_artist(elli)


    def plot_interest_points(cls, axes, current_frame, circle_coords, gaze_sampler, title="Sampled Interest Points (IP)"):
        """Plot Interest points.

        Args:
            axes (axes.Axes): Axes to plot the image
            current_frame (np.ndarray): Current frame
            circle_coords (np.ndarray): Coordinates of interest points
            title (str, optional): Title of the plot. Defaults to "Sampled Interest Points (IP)".
        """
        cls._image_plot(axes, current_frame, title)
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

    def plot_empirical_dists(cls, axes, hist_mat, title="IP Empirical Distribution"):
        """Plot IP Empirical distribution.

        Args:
            axes (axes.Axes): Axes to plot the image
            hist_mat (np.ndarray): 2D histogram array, already transposed to original
            title (str, optional): Title of the plot. Defaults to 'IP Empirical Distribution'.
        """
        cls._configure_axis(axes, title)
        sns.heatmap(hist_mat, linewidth=0.2, cbar=False, cmap='jet', ax=axes)



    def plot_order_disorder(cls, axes, order_series, disorder_series, lw=2):
        """Plot order/disorder.

        Args:
            axes (axes.Axes): Axes to plot the image
            order_series (vector): Order series vector.
            disorder_series (vector): Disorder series vector.
            lw (int, optional): Line width. Defaults to 2.
        """
        cls._configure_axis(axes, "Order/Disorder", keep_axis=True)
        axes.plot(disorder_series, 'r--', label='Disorder', linewidth=lw)
        axes.plot(order_series, 'g-', label='Order', linewidth=lw)
        axes.legend(loc='upper right')


    def plot_complexity(cls, axes, complexity_series, lw=2):
        """Plot complexity curves.

        Args:
            axes (axes.Axes): Axes to plot the image
            complexity_series (vector): Complexity series vector.
            lw (int, optional): Line width. Defaults to 2.
        """
        cls._configure_axis(axes, "Complexity" , keep_axis=True)
        axes.plot(complexity_series, label='Complexity', linewidth=lw)


    def create_result_folder(cls, results_folder, plot_type_folder):
        """Create result folder.

        Args:
            results_folder (str): Path to the results folder.
            plot_type_folder (str): Plot type folder.

        Returns:
            str: Path to the result folder.
        """
        # Create result folder
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        # Create subfolders
        if not os.path.exists(results_folder + '/' + plot_type_folder):
            os.makedirs(results_folder + '/' + plot_type_folder)

        return results_folder + '/' + plot_type_folder

    def plot_sampled_FOA(cls, axes, current_frame, foa_center, foa_radius, title="Final FOA"):
        cls._image_plot(axes, current_frame, title)
        foa_mask = cls._create_circular_mask(current_frame.shape[0], current_frame.shape[1], foa_center, foa_radius)
        cls._image_plot(axes, foa_mask, title,
                        cmap='gray', interpol='nearest')
#         if SAVE_FOA_IMG
#             [X,MAP]= frame2im(getframe);
#             FILENAME=[RESULT_DIR VIDEO_NAME '/FOA/FOA' imglist(iFrame).name];
#             imwrite(X,FILENAME,'jpeg');

    def _create_circular_mask(cls, h, w, center, radius):

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        mask = np.ma.masked_where(mask == 1, mask)

        return mask
