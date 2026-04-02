# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
"""

 =========================================================================
 CONCEPT 1: Camera-to-World vs. World-to-Camera
 =========================================================================
 The `transform_matrix` from the dataset is a Camera-to-World (C2W) matrix.
 It tells you where the camera is in the world.
 PyTorch3D needs `R` and `T` which define the World-to-Camera (W2C) transform
 (i.e., how to move world points into the camera's local frame).

 TODO: Convert the C2W matrices into W2C matrices.

 =========================================================================
 CONCEPT 2: Coordinate System Conventions (The tricky part!)
 =========================================================================
 Standard OpenGL/NeRF Camera Space:
   +X is Right
   +Y is Up
   +Z is Backward (the camera looks down the -Z axis)

 PyTorch3D Camera Space (NDC space):
   +X is Left
   +Y is Up
   +Z is Forward (the camera looks down the +Z axis)

 TODO: Create an adjustment matrix to flip the X and Z axes of your W2C
 matrices to match PyTorch3D's expected local camera coordinate system.

 =========================================================================
 CONCEPT 3: Row-Major vs Column-Major Transformation (R and T)
 =========================================================================
 Most computer vision math uses column vectors: P_cam = R @ P_world + T
 PyTorch3D uses row vectors: P_cam = P_world @ R + T

 Because of this, the 3x3 Rotation matrix `R` you pass to PyTorch3D must
 be the TRANSPOSE of the standard rotation matrix.
 `T` should be a shape (N, 3) tensor.

 TODO: Extract the properly transposed `R` (N, 3, 3) and `T` (N, 3)
 from your adjusted W2C matrices.

=========================================================================
CONCEPT 4: Intrinsics (Field of View / Focal Length)
=========================================================================
The dataset gives you `camera_angle_x`, which is the horizontal FOV in radians.
PyTorch3D's standard PerspectiveCameras expects a `focal_length`.
PyTorch3D uses Normalized Device Coordinates (NDC) by default, where the
shortest side of the image goes from -1 to 1.
Standard conversion from FOV to focal length is: f = 1.0 / tan(FOV / 2)
TODO: Calculate the focal length using `camera_angle_x`.
Note: If your images are square (like 800x800 in the Blender datasets),
fx == fy, so a single focal length value is fine.
=========================================================================
Finally, initialize the PyTorch3D cameras
=========================================================================
R = ... (Shape: N, 3, 3)
T = ... (Shape: N, 3)
focal_length = ... (Shape: N, 1 or N, 2)
cameras = PerspectiveCameras(
    R=R,
    T=T,
    focal_length=focal_length,
    image_size=[image_size] * N,  # Needed if you want to use screen coordinates later
    device=transform_matrices.device
)

# return cameras

"""
"""
get_world_to_view_transform
This function returns a Transform3d representing the transformation
matrix to go from world space to view space by applying a rotation and
a translation.

PyTorch3D uses the same convention as Hartley & Zisserman.
I.e., for camera extrinsic parameters R (rotation) and T (translation),
we map a 3D point `X_world` in world coordinates to
a point `X_cam` in camera coordinates with:
`X_cam = X_world R + T`

Args:
    R: (N, 3, 3) matrix representing the rotation.
    T: (N, 3) matrix representing the translation.

Returns:
    a Transform3d object which represents the composed RT transformation.

"""
# %%
import torch
import pytorch3d.io as p3di
import pytorch3d.structures as p3ds
import pytorch3d.renderer as p3dr
import pytorch3d.loss as p3dl
import pytorch3d.ops as p3do
import pytorch3d.utils as p3du
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import math
import dataloader
import utils
import recon_bench

from dataclasses import dataclass, field
from typing import Callable
from typing import Literal
from typing import Any

@dataclass
class LossConfig:
    """
    Configuration for a loss term with optional weight annealing.

    Attributes
    ----------
    weight : float
        The initial or constant weight of the loss term.
    end_weight : float | None, optional
        The final weight of the loss term after annealing. If None, the
        weight remains constant. Default is None.
    strategy : Literal["linear", "cosine", "exponential"], optional
        The annealing strategy to use. Default is "linear".
    loss_fn : Callable[[p3ds.Meshes], torch.Tensor] | None, optional
        The function used to compute the loss. Default is None.
    kwargs : dict[str, Any]
        Additional keyword arguments to pass to the loss function.
    values : list[float]
        A history of the computed loss values.
    weight_history : list[float]
        A history of the weights applied to the loss term.
    """
    weight: float
    end_weight: float | None = None
    strategy: Literal["linear", "cosine", "exponential"] = "linear"
    loss_fn: Callable[[p3ds.Meshes], torch.Tensor] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict) # pyright: ignore[reportAny] - kwargs type depends entirely on the downstream loss function
    values: list[float] = field(default_factory=list)
    weight_history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        Initialize the starting weight for annealing.
        """
        self.start_weight: float = self.weight

    def step(self, progress: float) -> None:
        """
        Update the weight according to the annealing strategy.

        Parameters
        ----------
        progress : float
            The current progress of the optimization, expected to be in
            the range [0.0, 1.0].
        """
        if self.end_weight is None:
            return
        match self.strategy:
            case "linear":
                t = progress
            case "cosine":
                t = (1 - math.cos(math.pi * progress)) / 2
            case "exponential":
                t = (math.exp(progress) - 1) / (math.e - 1)
            case _:
                raise ValueError(f"Unknown annealing strategy: {self.strategy}")

        self.weight = self.start_weight + (self.end_weight - self.start_weight) * t
        self.weight_history.append(self.weight)

# The optimizer requries the mesh for initialization, and the scheduler requries the optimizer, so this allows passing both in to the optimization loop.
type OptimizerFactory = Callable[[list[torch.Tensor]], torch.optim.Optimizer]
type SchedulerFactory = Callable[
    [torch.optim.Optimizer],
    torch.optim.lr_scheduler.LRScheduler,
]

# %% Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = pathlib.Path("./data")

# %% Define camera & mesh utils
def load_mesh(
        file_path: pathlib.Path,
        file_type: Literal["mesh", "pointcloud"],
        device: torch.device,
) -> p3ds.Meshes | p3ds.Pointclouds:
    """
    Load a 3D asset from disk as a mesh or point cloud.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the OBJ file.
    file_type : Literal["mesh", "pointcloud"]
        Whether to load as a triangle mesh or a point cloud.
    device : torch.device
        Device to place the loaded geometry on.

    Returns
    -------
    p3ds.Meshes | p3ds.Pointclouds
        The loaded geometry.
    """
    if file_type == "pointcloud":
        mesh = p3di.IO().load_pointcloud(file_path, device=device)
    else:
        mesh = p3di.load_objs_as_meshes([file_path], device=device)

    return mesh

def scale_norm_center_mesh(mesh: p3ds.Meshes) -> p3ds.Meshes:
    """
    Center a mesh at the origin and normalize it to fit within a unit cube.

    Parameters
    ----------
    mesh : p3ds.Meshes
        Input mesh to normalize. Modified in-place.

    Returns
    -------
    p3ds.Meshes
        The centered and scaled mesh.
    """
    verts = mesh.verts_packed()
    assert verts is not None
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center)
    mesh.scale_verts_((1.0 / float(scale)))

    return mesh

def create_silhouette_renderer(
    cameras: p3dr.PerspectiveCameras,
    image_size: int | tuple[int, int],
) -> p3dr.MeshRenderer:
    """
    Create a soft silhouette renderer.

    Uses ``SoftSilhouetteShader`` with blurred rasterization for
    differentiable silhouette rendering.

    Parameters
    ----------
    cameras : p3dr.PerspectiveCameras
        Cameras to bind to the rasterizer.
    image_size : int | tuple[int, int]
        Output image dimensions as a single int or (height, width).

    Returns
    -------
    p3dr.MeshRenderer
        Renderer that produces silhouette images of shape (N, H, W, 4).
    """
    # Rasterization settings for silhouette rendering
    sigma = 1e-4
    raster_settings_silhouette = p3dr.RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1. / 1e-4 - 1.)*sigma,
        faces_per_pixel=50,
    )

    # Silhouette renderer
    renderer_silhouette = p3dr.MeshRenderer(
        rasterizer=p3dr.MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings_silhouette
        ),
        shader=p3dr.SoftSilhouetteShader()
    )

    return renderer_silhouette

def create_vis_renderer(
    cameras: p3dr.PerspectiveCameras,
    image_size: int | tuple[int, int],
    lights: p3dr.PointLights,
) -> p3dr.MeshRenderer:
    """
    Create a Phong-shaded renderer for visualization.

    Parameters
    ----------
    cameras : p3dr.PerspectiveCameras
        Cameras to bind to the rasterizer.
    image_size : int | tuple[int, int]
        Output image dimensions as a single int or (height, width).
    lights : p3dr.PointLights
        Point light sources for Phong shading.

    Returns
    -------
    p3dr.MeshRenderer
        Renderer that produces RGB images of shape (N, H, W, 4).
    """
    raster_settings = p3dr.RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    renderer = p3dr.MeshRenderer(
        rasterizer=p3dr.MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        ),
        shader=p3dr.SoftPhongShader(
            device=DEVICE,
            cameras=cameras,
            lights=lights,
        )
    )

    return renderer

# %%
def soft_iou_loss(
    pred_silhouette: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Computes a differentiable IoU loss between a predicted soft silhouette and a ground-truth binary mask.

    Parameters
    ----------
    pred_silhouette : torch.Tensor
        Tensor of shape (B, H, W) with values in [0, 1].
    target_mask : torch.Tensor
        Tensor of shape (B, H, W) with values in {0, 1} (e.g., the alpha channel).

    Returns
    -------
    torch.Tensor
        A scalar loss value (1.0 - IoU). We minimize this, so a perfect match (IoU=1) gives a loss of 0.
    """
    # 1. Intersection: Element-wise multiplication
    # If both are high, intersection is high. If either is 0, intersection is 0.
    # This handles semi-transparent pixels produced by the differential renderer.
    intersection = pred_silhouette * target_mask

    # Sum over the spatial dimensions (H, W) to get the total intersection area per image
    intersection_area = intersection.sum(dim=(1, 2))

    # 2. Union: sum of both areas MINUS the intersection (to avoid double counting the overlap)
    pred_area = pred_silhouette.sum(dim=(1, 2))
    gt_area = target_mask.sum(dim=(1, 2))
    union_area = pred_area + gt_area - intersection_area

    # Add a tiny epsilon to the denominator to prevent division by zero if both images are totally empty
    eps = 1e-6
    iou = intersection_area / (union_area + eps)

    # 3. The Loss
    # Taking the mean across the batch (B) gives the final scalar loss.
    return 1.0 - iou.mean()

def calculate_silhouette_loss(
    targets: torch.Tensor,
    pred_mesh: p3ds.Meshes,
    batch_R: torch.Tensor,
    batch_T: torch.Tensor,
    cam_focal_length: torch.Tensor,
    method: Literal["iou", "mse"] = "iou",
) -> torch.Tensor:
    """
    Compute the loss between predicted and target silhouettes.

    Renders the mesh from each camera viewpoint and compares the alpha
    channel against the target images.

    Parameters
    ----------
    targets : torch.Tensor
        Ground-truth images of shape (N, C, H, W) in CHW format.
    pred_mesh : p3ds.Meshes
        Predicted meshes to render (batched).
    batch_R : torch.Tensor
        Camera rotation matrices of shape (N, 3, 3).
    batch_T : torch.Tensor
        Camera translation vectors of shape (N, 3).
    cam_focal_length : torch.Tensor
        Camera focal lengths.
    method : Literal["iou", "mse"], optional
        The loss computation method. Default is "iou".

    Returns
    -------
    torch.Tensor
        Scalar MSE loss over the alpha channel.
    """
    target_shape = targets[0].shape
    lights = p3dr.PointLights(device=DEVICE, location=[[0.0, 0.0, -3.0]])

    cameras = p3dr.PerspectiveCameras(
        focal_length = cam_focal_length,
        R = batch_R,
        T = batch_T,
        image_size=[(target_shape[-2], target_shape[-1])] * targets.size(0),
        device=DEVICE,
    )

    renderer = create_silhouette_renderer(
        cameras,
        (target_shape[-2], target_shape[-1]),
    )

    pred_images = renderer(pred_mesh, cameras=cameras, lights=lights)
    target_images = targets.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)

    if method == "mse":
        # Mean of MSE across batch
        error = ((pred_images[..., 3] - target_images[..., 3]) ** 2).mean()
    elif method == "iou":
        error = soft_iou_loss(pred_images[..., 3], target_images[...,3])
    else:
        raise ValueError(f"Unknown method {method}")

    return error

# %% Main optimization loop
def optimize_mesh(
    train_path: pathlib.Path,
    losses: dict[str, LossConfig],
    epochs: int = 100,
    batch_size: int = 4,
    optimizer_fn: OptimizerFactory = lambda params: torch.optim.SGD(params, lr=1.0, momentum=0.9),
    scheduler_fn: SchedulerFactory | None = None,
    plot_every: int | None = None,
    profiling: bool = False,
) -> tuple[p3ds.Meshes, dict[str, LossConfig], list[float]]:
    """
    Optimize a sphere mesh to match training images via silhouette fitting.

    Starts from an icosphere and iteratively deforms its vertices to minimize
    a weighted combination of silhouette, edge, normal, and laplacian losses.

    Parameters
    ----------
    train_path : pathlib.Path
        Path to the NeRF synthetic dataset transforms JSON.
    losses : dict[str, LossConfig]
        Mapping of loss name to its configuration. Each config's ``weight``,
        ``loss_fn``, and annealing strategy are used during training. Loss
        values and weight history are accumulated in-place.
    epochs : int
        Number of full passes over the dataset. Default is 100.
    batch_size : int
        Number of camera views per optimization step. Default is 4.
    optimizer_fn : OptimizerFactory
        Factory that receives a list of parameters and returns an optimizer.
        Default creates SGD with lr=1.0 and momentum=0.9.
    scheduler_fn : SchedulerFactory | None
        Factory that receives an optimizer and returns a learning rate
        scheduler. Called once per epoch. Default is None (no scheduling).
    plot_every : int | None
        Epoch interval for visualizing the mesh against the reference image.
        If None, no visualization is shown during training.
    profiling : bool
        Whether to enable performance profiling. Default is False.

    Returns
    -------
    tuple[p3ds.Meshes, dict[str, LossConfig], list[float]]
        The optimized deformed mesh, the losses dict with populated
        ``values`` and ``weight_history`` for each config, and the learning
        rate history.
    """
    dataset = dataloader.NeRFSyntheticDataset(train_path)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # Set up profiling
    timer = recon_bench.Timer(enabled=profiling)
    mem = recon_bench.MemoryTracker(enabled=profiling)


    # Set up static camera
    first_R, first_T, ref_image = dataset[0]
    first_R, first_T, ref_image = first_R.to(DEVICE), first_T.to(DEVICE), ref_image.to(DEVICE)
    ref_h, ref_w = ref_image.shape[-2], ref_image.shape[-1]
    cameras = p3dr.PerspectiveCameras(
        R=first_R.unsqueeze(0),
        T=first_T.unsqueeze(0),
        focal_length=dataset.focal_length,
        image_size=[(ref_h, ref_w)],
        device=DEVICE,
    )
    renderer_silhouette = create_silhouette_renderer(
        cameras,
        (ref_h, ref_w),
    )

    # Init source shape as a sphere
    src_mesh = p3du.ico_sphere(4, device=DEVICE)

    # Setup deformation mesh
    deform_mesh = src_mesh
    deform_verts = torch.full(
        src_mesh.verts_packed().shape,
        0.0,
        device=DEVICE,
        requires_grad=True
    )

    optimizer = optimizer_fn([deform_verts])
    scheduler = scheduler_fn(optimizer) if scheduler_fn is not None else None
    learning_rate = []

    def _run_batch(batch_idx, batch, train_pbar) -> torch.Tensor:
        with timer.section("batch"), mem.section("batch"):
            batch_loss = torch.tensor(0.0, device=DEVICE)
            R_matrices, T_vectors, target = batch
            R_matrices, T_vectors, target = R_matrices.to(DEVICE), T_vectors.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            deform_mesh = src_mesh.offset_verts(deform_verts)

            for key, config in losses.items():
                if key == "silhouette":
                    loss_silhouette = calculate_silhouette_loss(
                        targets=target,
                        pred_mesh=deform_mesh.extend(R_matrices.shape[0]),
                        batch_R=R_matrices,
                        batch_T=T_vectors,
                        cam_focal_length=dataset.focal_length,
                        **config.kwargs,
                    )
                    loss_silhouette /= batch_size
                    config.values.append(loss_silhouette.item())
                    batch_loss += loss_silhouette * config.weight
                elif config.loss_fn is not None:
                    loss = config.loss_fn(deform_mesh)
                    config.values.append(loss.item())
                    batch_loss += loss * config.weight

            train_pbar.set_description(f"Batch {batch_idx} loss: {batch_loss:.4f}")

            with timer.section("backward"), mem.section("backward"):
                batch_loss.backward()
                # Clip grads to avoid large vertex changes
                torch.nn.utils.clip_grad_norm_([deform_verts], max_norm=1.0)
                optimizer.step()

            return batch_loss.detach()

    def _run_epoch(i, epoch_loop):
        with timer.section("epoch"), mem.section("epoch"):
            epoch_loss = torch.tensor(0.0, device=DEVICE)

            train_pbar = tqdm.tqdm(train_loader, desc="Batch Optimization", leave=False)
            for batch_idx, batch in enumerate(train_pbar):
                epoch_loss += _run_batch(batch_idx, batch, train_pbar)

            epoch_loss /= len(train_loader)
            current_lr = optimizer.param_groups[0]['lr']
            epoch_loop.set_description(f"Epoch {i}/{epochs} loss: {epoch_loss:.6f} lr: {current_lr:.4e}")

            # Update the LR scheduler if needed
            if scheduler is not None:
                learning_rate.append(current_lr)
                scheduler.step()

            # Update the LossConfig
            progress = (i + 1) / epochs
            for config in losses.values():
                config.step(progress)

            if plot_every is not None and i % plot_every == 0:
                utils.visualize_prediction(
                    src_mesh.offset_verts(deform_verts),
                    renderer=renderer_silhouette,
                    target_image=ref_image.permute(1, 2, 0), # (C, H, W) -> (H, W, C)
                    title=f"Epoch: {i}",
                    silhouette=True,
                )
                plt.show(block=False)

    with timer.section("train"), mem.section("train"):
        epoch_loop = tqdm.tqdm(range(epochs), desc="Optimizing")
        for i in epoch_loop:
            _run_epoch(i, epoch_loop)

    deform_mesh = src_mesh.offset_verts(deform_verts)

    # Visualize final step
    utils.visualize_prediction(
        deform_mesh,
        renderer=renderer_silhouette,
        target_image=ref_image.permute(1, 2, 0), # (C, H, W) -> (H, W, C)
        title="Final mesh:",
        silhouette=True,
    )
    plt.show()

    # Build a report
    report = recon_bench.ProfileResult(
        timing=timer.get_report(),
        memory=mem.get_report(),
        cuda_available=mem.cuda_available,
    )
    print(report.summary())

    return deform_mesh, losses, learning_rate


# %% Setup Hypers
# Build weighted losses with annealing
loss_configs: dict[str, LossConfig] = {
    "silhouette": LossConfig(
        weight=1.0,
        kwargs={"method": "iou"},
    ),
    "edge": LossConfig(
        weight=0.1,
        end_weight=1.0,
        strategy="cosine",
        loss_fn=p3dl.mesh_edge_loss,
    ),
    "normal": LossConfig(
        weight=0.005,
        end_weight=0.01,
        strategy="linear",
        loss_fn=p3dl.mesh_normal_consistency,
    ),
    "laplacian": LossConfig(
        weight=0.1,
        end_weight=0.6,
        strategy="exponential",
        loss_fn=p3dl.mesh_laplacian_smoothing,
    ),
}
num_epochs = 1000
views_per_iteration = 3
plot_freq = 25

# %% Run optimization
new_mesh, losses, learning_rate = optimize_mesh(
    pathlib.Path("../../data/ficus/transforms_train.json"),
    losses=loss_configs,
    epochs=num_epochs,
    batch_size=views_per_iteration,
    optimizer_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
    scheduler_fn=lambda opt: torch.optim.lr_scheduler.StepLR(
        opt,
        step_size=250,
        gamma=0.99
    ),
    plot_every=plot_freq,
    profiling=False,
)

# %%
import utils
import importlib

importlib.reload(utils)

# %% Plot losses
utils.plot_losses(losses)
plt.show()
utils.plot_weight_history(losses, learning_rate)
plt.show()


# %% Export mesh
new_mesh = scale_norm_center_mesh(new_mesh.detach())
final_verts, final_faces = new_mesh.get_mesh_verts_faces(0)

final_obj = pathlib.Path('final_model.obj')
p3di.save_obj(final_obj, final_verts, final_faces)
