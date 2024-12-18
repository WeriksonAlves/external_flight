from functools import wraps
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import List, Optional, Tuple, Type
import matplotlib.pyplot as plt
import numpy as np
import rospy


def validate_ndarray(func):
    """
    Decorator to validate if all positional arguments are NumPy ndarrays.
    Raises ValueError if validation fails.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        for arg in args:
            if not isinstance(arg, np.ndarray):
                raise ValueError(f"Expected np.ndarray, got {type(arg)}")
        return func(*args, **kwargs)
    return wrapper


class MyGraphics:
    """
    A utility class for generating plots related to tracked trajectories, PCA
    results, and confusion matrices.
    """

    @staticmethod
    @validate_ndarray
    def tracked_xyz(
        tracked_poses: np.ndarray, figsize: Tuple[int, int] = (8, 6)
    ) -> None:
        """
        Plots the smoothed trajectory of tracked points using their X and Y
        coordinates.

        :param tracked_poses: NumPy array containing tracked points.
                                     Must have at least 5 columns for X, Y,
                                     and additional coordinates.
        :param figsize: Tuple specifying the plot's dimensions (width, height).
        :raises ValueError: If the input array has fewer than 5 columns.
        """
        if tracked_poses.shape[1] < 5:
            raise ValueError(
                "Input array must have at least 5 columns for X, Y, and other"
                " coordinates."
            )

        rospy.loginfo("Plotting smoothed trajectory of tracked points.")
        plt.figure(figsize=figsize)
        plt.title('Smoothed Trajectory of Tracked Points')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.grid(True)

        # Plot trajectories
        plt.plot(
            tracked_poses[:, 0], tracked_poses[:, 1], 'ro-',
            label='Trajectory 1 (X, Y)'
        )
        plt.plot(
            tracked_poses[:, 3], tracked_poses[:, 4], 'bx-',
            label='Trajectory 2 (X, Y)'
        )
        plt.legend()
        plt.show()

    @staticmethod
    def results_pca(
        pca_model: Type[PCA], n_components: int = 3,
        figsize: Tuple[int, int] = (8, 6)
    ) -> None:
        """
        Visualizes the explained variance ratio per principal component.

        :param pca_model: A fitted PCA model instance.
        :param n_components: Number of components to visualize.
        :param figsize: Tuple specifying the plot's dimensions (width, height).
        :raises ValueError: If the PCA model is not fitted or `n_components`
                            exceeds available components.
        """
        if not hasattr(pca_model, 'explained_variance_ratio_'):
            raise ValueError("PCA model must be fitted before visualization.")
        if n_components > len(pca_model.explained_variance_ratio_):
            raise ValueError(
                "Number of components exceeds available components in PCA "
                "model."
            )

        explained_variance = pca_model.explained_variance_ratio_
        rospy.loginfo("Visualizing PCA explained variance ratio.")

        plt.figure(figsize=figsize)
        plt.bar(
            range(1, n_components + 1), explained_variance[:n_components],
            alpha=0.5, align='center', label='Variance'
        )
        plt.step(
            range(1, n_components + 1),
            np.cumsum(explained_variance[:n_components]),
            where='mid', label='Cumulative Variance'
        )
        plt.xlabel('Principal Components')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance Per Principal Component')
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def plot_confusion_matrix(
        y_true: List[str], y_predict: List[str], target_names: List[str],
        file_path: str, figsize: Tuple[int, int] = (8, 6)
    ) -> None:
        """
        Generates and saves confusion matrices (both normalized and absolute).

        :param y_true: List of true labels.
        :param y_predict: List of predicted labels.
        :param target_names: List of class names.
        :param file_path: Path to save the confusion matrix images.
        :param figsize: Tuple specifying the plot's dimensions (width, height).
        """
        rospy.loginfo("Generating confusion matrices.")
        cm_percentage = confusion_matrix(
            y_true, y_predict, labels=target_names, normalize='true'
        )
        cm_absolute = confusion_matrix(y_true, y_predict, labels=target_names)

        # Plot normalized confusion matrix
        plt.figure(figsize=figsize)
        ConfusionMatrixDisplay(
            confusion_matrix=cm_percentage, display_labels=target_names
        ).plot(cmap='Blues')
        plt.title('Normalized Confusion Matrix')
        plt.savefig(f"{file_path}_percentage.jpg")
        rospy.loginfo(
            f"Normalized confusion matrix saved at {file_path}_percentage.jpg"
        )
        plt.close()

        # Plot absolute confusion matrix
        plt.figure(figsize=figsize)
        ConfusionMatrixDisplay(
            confusion_matrix=cm_absolute, display_labels=target_names
        ).plot(cmap='Blues')
        plt.title('Absolute Confusion Matrix')
        plt.savefig(f"{file_path}_absolute.jpg")
        rospy.loginfo(
            f"Absolute confusion matrix saved at {file_path}_absolute.jpg"
        )
        plt.close()

    @staticmethod
    def set_axes_equal(ax: Axes3D) -> None:
        """
        Adjusts the scaling of a 3D plot so that all axes have equal range,
        maintaining proper aspect ratio.

        :param ax: The 3D axes object to adjust.
        """
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        # Calculate ranges and midpoints
        ranges = {
            "x": abs(x_limits[1] - x_limits[0]),
            "y": abs(y_limits[1] - y_limits[0]),
            "z": abs(z_limits[1] - z_limits[0]),
        }
        max_range = max(ranges.values())

        midpoints = {
            "x": np.mean(x_limits),
            "y": np.mean(y_limits),
            "z": np.mean(z_limits),
        }

        # Update axes limits
        for axis, midpoint in midpoints.items():
            ax.set_xlim3d(
                [midpoints["x"] - max_range / 2,
                 midpoints["x"] + max_range / 2])
            ax.set_ylim3d(
                [midpoints["y"] - max_range / 2,
                 midpoints["y"] + max_range / 2])
            ax.set_zlim3d(
                [midpoints["z"] - max_range / 2,
                 midpoints["z"] + max_range / 2])

    @staticmethod
    def create_3d_plot() -> Tuple[plt.Figure, Axes3D]:
        """
        Creates a 3D plot with a specified figure size.

        :return: A tuple containing the figure and axes objects.
        """
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        return fig, ax

    @staticmethod
    def plot_trajectory(trajectory_data: List[Tuple[float, float, float]],
                        fig: Optional[plt.Figure] = None, ax: Optional[
                            Axes3D] = None) -> None:
        """
        Visualizes the 3D trajectory of the drone based on the given data.

        :param trajectory_data: A list of (x, y, z) tuples representing the
                                drone's trajectory.
        :param fig: The figure object to use for plotting.
        :param ax: The 3D axes object to use for plotting.
        """
        if not trajectory_data:
            raise ValueError("Trajectory data cannot be empty.")

        # Extract coordinates
        x, y, z = zip(*trajectory_data)

        # Create 3D plot
        if not fig or not ax:
            fig, ax = MyGraphics.create_3d_plot()

        # Plot trajectory
        ax.scatter(x, y, z, color="b", label="Trajectory", s=20)
        ax.plot(x, y, z, color="r", alpha=0.6)

        # Set labels and titles
        ax.set_title("Drone Trajectory", fontsize=16)
        ax.set_xlabel("X (meters)", fontsize=12)
        ax.set_ylabel("Y (meters)", fontsize=12)
        ax.set_zlabel("Z (meters)", fontsize=12)
        ax.legend(fontsize=10)

        # Apply equal axes scaling and grid
        MyGraphics.set_axes_equal(ax)
        ax.grid(True)
