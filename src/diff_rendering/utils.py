from __future__ import annotations

import torch
import pytorch3d.renderer as p3dr
import pytorch3d.ops as p3do
import pytorch3d.structures as p3ds
import matplotlib.pyplot as plt
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import fit_mesh

def image_grid(
    images: np.ndarray,
    rows: int | None = None,
    cols: int | None = None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
) -> None:
    """
    Plot a grid of RGBA images using matplotlib.

    Parameters
    ----------
    images : np.ndarray of shape (N, H, W, 4)
        RGBA images to display.
    rows : int, optional
        Number of rows in the grid. Must be specified together with `cols`.
    cols : int, optional
        Number of columns in the grid. Must be specified together with `rows`.
    fill : bool, optional
        If True, remove spacing between subplots. Default is True.
    show_axes : bool, optional
        If True, show axis ticks and labels. Default is False.
    rgb : bool, optional
        If True, render only RGB channels. If False, render only the alpha
        channel. Default is True.

    Returns
    -------
    None
    """
    # ⚠️ WARNING: rows and cols must both be set or both be omitted
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    figure, axes = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    figure.subplots_adjust(left=0, bottom=0, right=1, top=1)

    for axis, image in zip(axes.ravel(), images):
        if rgb:
            axis.imshow(image[..., :3])  # 📝 NOTE: render RGB channels only
        else:
            axis.imshow(image[..., 3])   # 📝 NOTE: render alpha channel only
        if not show_axes:
            axis.set_axis_off()

def plot_pointcloud(mesh: p3ds.Meshes, title: str = "") -> None:
    """
    Plot a 3D scatter plot of points sampled from a mesh surface.

    Samples 5000 points from the mesh and displays them in a left-handed
    coordinate system with Z pointing inward.

    Parameters
    ----------
    mesh : p3ds.Meshes
        Source mesh to sample points from.
    title : str
        Plot title. Default is empty.
    """
    points = p3do.sample_points_points(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, z, -y)  # LH z-in
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.set_title(title)
    ax.view_init(190, 30)


def plot_losses(losses: dict[str, fit_mesh.LossConfig]) -> None:
    """
    Plot loss curves for each loss component over training iterations.

    Parameters
    ----------
    losses : dict[str, fit_mesh.LossConfig]
        Mapping of loss name to its config. Each config's ``values``
        attribute contains per-iteration loss floats.
    """
    figure = plt.figure(figsize=(13, 5))
    axis = figure.gca()

    for k, l in losses.items():
        axis.plot(l.values, label=k + " loss")

    axis.set_xlabel("Iteration", fontsize="16")
    axis.set_ylabel("Loss", fontsize="16")
    axis.set_title("Loss vs iterations", fontsize="16")

    # Optional: the losses themselves have vastly different scales
    axis.set_yscale("log")

    # Plot the standard legend
    axis.legend(fontsize="16")



def plot_weight_history(losses: dict[str, fit_mesh.LossConfig], learning_rate: list[float] | None = None) -> None:
    """
    Plot loss weight annealing curves over training epochs.

    Parameters
    ----------
    losses : dict[str, fit_mesh.LossConfig]
        Mapping of loss name to its config. Each config's ``weight_history``
        attribute contains per-epoch weight floats.
    learning_rate : list[float] | None, optional
        Learning rate values. If provided, a second y-axis
        will be plotted for the learning rate. Default is None.
    """
    figure = plt.figure(figsize=(13, 5))
    axis = figure.gca()
    for k, config in losses.items():
        if config.weight_history:
            axis.plot(config.weight_history, label=k)

    axis.set_xlabel("Epoch", fontsize="16")
    axis.set_ylabel("Weight", fontsize="16")
    axis.set_title("Loss weights vs epochs", fontsize="16")

    if learning_rate is not None:
        axis2 = axis.twinx()
        axis2.plot(learning_rate, label="learning rate", linestyle="--")
        axis2.set_ylabel("Learning rate", fontsize="16")

        # Combine legends from both axes
        lines_1, labels_1 = axis.get_legend_handles_labels()
        lines_2, labels_2 = axis2.get_legend_handles_labels()
        axis2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize="16")
    else:
        axis.legend(fontsize="16")


def visualize_prediction(
    predicted_mesh: p3ds.Meshes,
    renderer: p3dr.MeshRenderer,
    target_image: torch.Tensor,
    title: str = "",
    silhouette: bool = False,
) -> None:
    """
    Display a side-by-side comparison of a rendered prediction and a target image.

    Parameters
    ----------
    predicted_mesh : p3ds.Meshes
        Mesh to render for the prediction subplot.
    renderer : p3dr.MeshRenderer
        Renderer used to produce the predicted image.
    target_image : torch.Tensor
        Ground-truth image of shape (H, W, C).
    title : str
        Plot title. Default is empty.
    silhouette : bool
        If True, display only the alpha channel (index 3). If False, display
        RGB channels (indices 0-2). Default is False.
    """
    channels: int | range = 3 if silhouette else range(3)
    with torch.no_grad():
        predicted_images = renderer(predicted_mesh)
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(predicted_images[0, ..., channels].detach().cpu().numpy())

    plt.subplot(1, 2, 2)
    plt.imshow(target_image.detach().cpu().numpy())
    plt.title(title)
    plt.axis("off")
