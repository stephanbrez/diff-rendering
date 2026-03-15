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

        # List of all frames containing image paths and poses
        self.frames = self.meta['frames']

        # PyTorch standard transformation
        self.transform = ToTensor()

    def __len__(self) -> int:
        """Return the number of frames in the dataset."""
        return len(self.frames)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load a single frame's camera pose and image.

        Parameters
        ----------
        idx : int
            Frame index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A pair of (transform_matrix, img_tensor) where:
            - transform_matrix has shape (4, 4), the camera-to-world matrix.
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

        # 3. Get the camera pose
        # The transform_matrix is a 4x4 camera-to-world matrix
        transform_matrix = torch.tensor(frame['transform_matrix'], dtype=torch.float32)

        # Return the pair (Transform, Image)
        return transform_matrix, img_tensor
