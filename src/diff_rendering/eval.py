# %%
import recon_bench
import math
import torch
import dataloader
import pathlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
dataset = dataloader.NeRFSyntheticDataset('../../data/ficus/transforms_train.json')
dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=4)

R_matrices, T_vectors, targets = next(iter(dataloader))
R_matrices, T_vectors, targets = R_matrices.to(device), T_vectors.to(device), targets.to(device)
targets_list = list((targets.split(1, dim=0)))

# %%
# Convert from pytorch3D > open3d
# Camera:
# X - left, Y - up, Z - forward > X - right, Y - up, Z - backwards
# World:
# X - left, Y - up, Z - forward > X - right, Y - forward, Z - up

P_cam = torch.tensor([
    [-1,  0,  0, ],
    [ 0, -1,  0, ],
    [ 0,  0,  1, ],
], dtype=torch.float32, device=device)


R_matrices = R_matrices.transpose(-1, -2)
R_matrices = P_cam @ R_matrices

# # Fix batched T_vectors
T_vectors = (P_cam @ T_vectors[..., None])[..., 0]

# 800x800 images so fov_x = fov_y
# %%
fov_y = math.degrees(dataset.camera_angle_x)

fy_o3d = (targets.shape[-2] / 2) / math.tan(math.radians(fov_y) / 2)
fx_o3d = fy_o3d

fx_from_dataset = dataset.focal_length.item() * (targets.shape[-1] / 2)

print(dataset.camera_angle_x, fov_y)
print("Open3D fx/fy px:", fx_o3d, fy_o3d)
print("Dataset focal converted to px:", fx_from_dataset)

# %%
fov_y = math.degrees(dataset.camera_angle_x)
cameras = [
    recon_bench.Camera.from_extrinsics(
        R_matrix,
        T_vector,
        width=targets_list[i].shape[3],
        height=targets_list[i].shape[2],
        fov=fov_y,
    )
    for i, (R_matrix, T_vector) in enumerate(zip(R_matrices, T_vectors))
]

# %%
results = recon_bench.evaluate(targets_list, pathlib.Path("final_model_iou_b3_level5_exp.obj"), camera=cameras)

# %%
results.summary()

# %%
results.detail()

# %%
results.save_renders(pathlib.Path('../../output/'))
results.save_targets(pathlib.Path('../../output/'))
# %%
