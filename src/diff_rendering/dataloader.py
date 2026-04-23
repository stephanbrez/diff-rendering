"""
Dataset loaders for differentiable rendering with PyTorch3D.

This module provides PyTorch Dataset implementations for loading image data
and camera poses from various sources, such as Blender-formatted NeRF synthetic
datasets and COLMAP outputs. It handles the critical conversion of camera
extrinsics and coordinate systems to match PyTorch3D's expected formats.

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

"""
import os
import json
import torch
import pathlib
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image

class NeRFSyntheticDataset(Dataset):
    def __init__(self, json_path, img_dir=None):
        """
        Initialize a NeRF synthetic dataset.

        Loads camera poses and image paths from a Blender-format transforms
        JSON file (e.g. ficus, chair, hotdog).

        Parameters
        ----------
        json_path : pathlib.Path
            Path to the transforms JSON file (e.g. ``transforms_train.json``).
        img_dir : str, optional
            Base directory for images. If None, image paths in the JSON are
            resolved relative to the JSON file's directory.
        """
        self.json_path = pathlib.Path(json_path)
        self.base_dir = self.json_path.parent if img_dir is None else pathlib.Path(img_dir)

        with open(self.json_path, 'r') as f:
            self.meta = json.load(f)

        # The camera horizontal field of view is stored in the json
        self.camera_angle_x = self.meta.get('camera_angle_x', None)
        self.focal_length = 1.0 / torch.tan(torch.tensor(self.camera_angle_x / 2.0)) # NDC width/2 = 1.0

        # List of all frames containing image paths and poses
        self.frames = self.meta['frames']

        # PyTorch standard transformation
        self.transform = ToTensor()

        # Convert from NeRF camera poses to PyTorch3D format
        c2w_matrices = torch.tensor([frame['transform_matrix'] for frame in self.frames], dtype=torch.float32)
        # Convert the C2W matrices into W2C matrices.
        w2c_matrices = torch.linalg.inv(c2w_matrices)

        # Camera space: flip X and Z - Blender -> Pytorch3d
        P_cam = torch.tensor([
            [-1,  0,  0,  0],
            [ 0,  1,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  0,  0,  1],
        ], dtype=torch.float32,)

        # World spaces: swap Y and Z
        P_world = torch.tensor([
            [1,  0,  0,  0],
            [0,  0,  1,  0],
            [0,  1,  0,  0],
            [0,  0,  0,  1],
        ], dtype=torch.float32,)

        # Apply both in one step
        w2c_p3d = P_cam @ w2c_matrices @ P_world

        self.R = w2c_p3d[:, :3, :3]
        self.R = self.R.transpose(1, 2) # switch to row-major
        self.T = w2c_p3d[:, :3, 3]

    def __len__(self) -> int:
        """Return the number of frames in the dataset."""
        return len(self.frames)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load a single frame's camera pose and image.

        Parameters
        ----------
        idx : int
            Frame index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple of (rotation matrix, translation vector, img_tensor) where:
            - rotation matrix has shape (3, 3).
            - translation vector has shape (3,).
            - img_tensor has shape (4, H, W), the RGBA image in CHW format
              with pixel values normalized to [0.0, 1.0].
        """
        frame = self.frames[idx]

        # 1. Get the image path
        rel_path = frame['file_path']

        # The Blender synthetic dataset JSONs often omit the '.png' extension
        if not rel_path.endswith('.png'):
            rel_path += '.png'

        # If the path starts with './', strip it so os.path.join works properly if img_dir was overridden
        if rel_path.startswith('./'):
            rel_path = rel_path[2:]

        full_img_path = os.path.join(self.base_dir, rel_path)

        # 2. Load the image
        # Images are RGBA (4 channels), usually with a transparent background
        img = Image.open(full_img_path).convert("RGBA")

        # ToTensor() converts HWC -> CHW and normalizes pixels to [0.0, 1.0]
        img_tensor = self.transform(img)

        # Return the converted components
        return self.R[idx], self.T[idx], img_tensor


class COLMAPDataset(Dataset):
    def __init__(self):
        w2c_matrices = transform_matrices # no conversion

        # Camera space: flix X and Y - COLMAP -> Pytorch3d
        P_cam = torch.tensor([
            [-1, 0,  0,  0],
            [0, -1,  0,  0],
            [0,  0,  1,  0],
            [0,  0,  0,  1],
        ], dtype=torch.float32,)

        # World spaces: no flip needed

        # Apply
        w2c_p3d = P_cam @ w2c_matrices

        self.R = w2c_p3d[:, :3, :3]
        self.R = self.R.transpose(1, 2) # switch to row-major
        self.T = w2c_p3d[:, :3, 3]

    def __len__(self):
        return len(self.frames)

    def __get_item__(self):
        # If the path starts with './', strip it so os.path.join works properly if img_dir was overridden
        if rel_path.startswith('./'):
            rel_path = rel_path[2:]

        full_img_path = os.path.join(self.base_dir, rel_path)

        # 2. Load the image
        # Images are RGBA (4 channels), usually with a transparent background
        img = Image.open(full_img_path).convert("RGBA")

        # ToTensor() converts HWC -> CHW and normalizes pixels to [0.0, 1.0]
        img_tensor = self.transform(img)

        # Return the converted components
        return self.R[idx], self.T[idx], img_tensor
