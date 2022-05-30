import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle


class Plotter:

    def __init__(self, num_line=2, num_pics_line=5, fig_size=(15, 5)):
        fig, ax = plt.subplots(num_line, num_pics_line, figsize=fig_size)
        self.fig = fig
        self.axes = ax
        plt.ion()
        plt.show()

    def _clean_axes(self):
        for i in range(2):
            for j in range(5):
                self.axes[i, j].clear()

    @classmethod
    def _configure_axis(cls, ax, title, keep_axis=False):
        if not keep_axis:
            ax.set_axis_off()
        ax.set_title(title)

    @classmethod
    def _image_plot(cls, ax, image, title, cmap=None, interpol=None):
        cls._configure_axis(ax, title)
        ax.imshow(image, cmap=cmap, interpolation=interpol)


    def plot_visualization(self, data, pause_time=0.001):
        # if GeneralConfig.VISUALIZE_RESULTS:
        self._clean_axes()
        self.plot_original_frame(self.axes[0, 0], data["original_frame"])
        self.plot_foveated_frame(self.axes[0, 1], data["foveated_frame"])
        self.plot_feature_map(self.axes[0, 2], data["feature_map"])
        self.plot_saliency_map(self.axes[0, 3], data["saliency_map"])
        self.plot_proto_objects(self.axes[0, 4], data["original_frame"],
                                data["proto_mask"], data["num_proto"],
                                data["proto_params"], data["nV"])
        self.plot_interest_points(self.axes[1, 0], data["original_frame"],
                                  data["circle_coords"])
        self.plot_empirical_dists(self.axes[1, 1], data["hist_mat"])
        self.plot_order_disorder(self.axes[1, 2], data["order"], data["disorder"])
        self.plot_complexity(self.axes[1, 3], data["complexity"])

        plt.tight_layout()
        plt.pause(pause_time)

    def plot_original_frame(cls, axes, original_frame):
        #     # 1. The original frame.
        cls._image_plot(axes, original_frame, 'Current frame')

    def plot_foveated_frame(cls, axes, foveated_frame):
        # 2. The foveated frame.
        cls._image_plot(axes, foveated_frame, 'Foveated frame', cmap='gray')
    #     if GeneralConfig.SAVE_FOV_IMG:
    #         pass
    #         # [X,MAP]= frame2im(getframe);
    #         # FILENAME=[RESULT_DIR VIDEO_NAME '/FOV/FOV' imglist(iFrame).name];
    #         # imwrite(X,FILENAME,'jpeg');

    def plot_feature_map(cls, axes, feature_map):
        # 3. The feature map
        cls._image_plot(axes, feature_map, 'Feature Map', cmap='gray')

    def plot_saliency_map(cls, axes, saliency_map):
        # 4. The saliency map
        cls._image_plot(axes, saliency_map, 'Saliency Map', cmap='jet')

        #     if GeneralConfig.SAVE_SAL_IMG:
        #         pass
        #         # [X,MAP]= frame2im(getframe);
        #         # FILENAME=[RESULT_DIR VIDEO_NAME '/SAL/SAL' imglist(iFrame).name];
        #         # imwrite(X,FILENAME,'jpeg');

    def plot_proto_objects(cls, axes, current_frame, proto_mask, num_proto, proto_params, nV):
        if num_proto > 0:
            cls._image_plot(axes, current_frame, 'Proto-Objects')
            cls._image_plot(axes, proto_mask, 'Proto-Objects',
                            cmap='gray', interpol='nearest')

            for p in range(nV):
                ((centx,centy), (width,height), angle) = proto_params["a"][p]
                elli = Ellipse((centx,centy), width, height, angle)
                elli.set_ec('yellow')
                elli.set_fill(False)
                axes.add_artist(elli)
                # if GeneralConfig.SAVE_PROTO_IMG:
                #     pass
                #         [X,MAP]= frame2im(getframe);
                #         FILENAME=[RESULT_DIR VIDEO_NAME '/PROTO/PROTO' imglist(iFrame).name];
                #         imwrite(X,FILENAME,'jpeg');

    def plot_interest_points(cls, axes, current_frame, circle_coords):
        #  6. The Interest points
        cls._image_plot(axes, current_frame, "Sampled Interest Points (IP)")
        # Show image with region marked
        xCoord, yCoord = circle_coords
        for b in range(yCoord.shape[0]):
            circle = Circle((xCoord[b],yCoord[b]), 4, color='r', lw=1)
            axes.add_artist(circle)

        #     # for idc in range(len(candx)):
        #     #     circle = Circle((candx[idc], candy[idc]), 4, color='y', lw=2)
        #     #     ax[1, 0].add_artist(circle)
        #     #     # drawcircle(candx(idc), candy(idc),4,'y',2);

        #     # circle = Circle((candidate_FOA[0], candidate_FOA[1]), 10, color='g', lw=6)
        #     # ax[1, 0].add_artist(circle)


        #     if GeneralConfig.SAVE_IP_IMG:
        #         pass
        #         # [X,MAP]= frame2im(getframe);
        #         # FILENAME=[RESULT_DIR VIDEO_NAME '/IP/IP' imglist(iFrame).name];
        #         # imwrite(X,FILENAME,'jpeg');

    def plot_empirical_dists(cls, axes, hist_mat):
        # 7. The IP Empirical distribution for computing complexity
        cls._configure_axis(axes, "IP Empirical Distribution")
        sns.heatmap(hist_mat, linewidth=0.2, cbar=False, cmap='jet', ax=axes)
        # if GeneralConfig.SAVE_HISTO_IMG:
        #     pass
            # [X,MAP]= frame2im(getframe);
            # FILENAME=[RESULT_DIR VIDEO_NAME '/HISTO/HISTO' imglist(iFrame).name];
            # imwrite(X,FILENAME,'jpeg');


    def plot_order_disorder(cls, axes, order_series, disorder_series, lw=2):
        #  8. The Complexity curves
        cls._configure_axis(axes, "Order/Disorder", keep_axis=True)
        axes.plot(order_series, 'r--', label='Disorder', linewidth=lw)
        axes.plot(disorder_series, 'g-', label='Order', linewidth=lw)


    def plot_complexity(cls, axes, complexity_series, lw=2):
        # 9. The Complexity curves
        cls._configure_axis(axes, "Complexity" , keep_axis=True)
        axes.plot(complexity_series, label='Complexity', linewidth=lw)
