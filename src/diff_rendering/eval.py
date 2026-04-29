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
results = recon_bench.evaluate(
    targets_list,
    pathlib.Path("final_model_optimized.obj"),
    camera=cameras,
    background_color=(0.0, 0.0, 0.0)
)

# %%
results.summary()

# %%
results.detail()

# %%
results.save_renders(pathlib.Path('../../output/'))
results.save_targets(pathlib.Path('../../output/'))

# %%
# Cross-check: render the same obj with PyTorch3D using the same source poses
# (recover P3D-format R/T from the OpenCV-converted ones; P_cam is involutive).
import numpy as np
import pytorch3d.io as p3di
import pytorch3d.renderer as p3dr
import pytorch3d.structures as p3ds
from PIL import Image

R_p3d = (P_cam @ R_matrices).transpose(-1, -2)[:1]
T_p3d = (P_cam @ T_vectors[..., None])[..., 0][:1]
H, W = targets_list[0].shape[2], targets_list[0].shape[3]

# Identify the frame: this cell renders batch item 0, which corresponds to
# the FIRST frame printed by NeRFSyntheticDataset during the earlier
# `next(iter(dataloader))` call. Use that frame's transform_matrix as the
# JSON ground-truth to compare against from_extrinsics' world-space output.
print(f"[p3d cross-check] using batch item 0; R_p3d=\n{R_p3d}\nT_p3d={T_p3d}")

verts, faces_idx, _ = p3di.load_obj("final_model_optimized.obj", device=device)
verts_rgb = torch.ones_like(verts)[None]
mesh = p3ds.Meshes(
    verts=[verts],
    faces=[faces_idx.verts_idx],
    textures=p3dr.TexturesVertex(verts_features=verts_rgb),
)

p3d_cameras = p3dr.PerspectiveCameras(
    R=R_p3d,
    T=T_p3d,
    focal_length=dataset.focal_length,
    image_size=[(H, W)],
    device=device,
)

raster_settings = p3dr.RasterizationSettings(
    image_size=(H, W),
    blur_radius=0.0,
    faces_per_pixel=1,
)
lights = p3dr.PointLights(device=device, location=p3d_cameras.get_camera_center())
blend_params = p3dr.BlendParams(background_color=(0.0, 0.0, 0.0))
phong_renderer = p3dr.MeshRenderer(
    rasterizer=p3dr.MeshRasterizer(cameras=p3d_cameras, raster_settings=raster_settings),
    shader=p3dr.SoftPhongShader(
        device=device,
        cameras=p3d_cameras,
        lights=lights,
        blend_params=blend_params,
    ),
)

rendered = phong_renderer(mesh)  # (1, H, W, 4)
rgb = rendered[0, ..., :3].detach().cpu().numpy()
img_uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(img_uint8).save("../../output/prediction_p3d_0.png")

# %%
# Quantify the scale gap: count foreground pixels in target vs the two renders,
# and report the linear scale ratio = sqrt(area_target / area_render).
def _foreground_mask(path: str, threshold: int = 10) -> np.ndarray:
    img = Image.open(path)
    if img.mode == "RGBA":
        rgb = Image.new("RGB", img.size, (0, 0, 0))
        rgb.paste(img, mask=img.split()[3])
        arr = np.asarray(rgb)
    else:
        arr = np.asarray(img.convert("RGB"))
    return arr.sum(axis=-1) > threshold

paths = {
    "target":         "../../output/target_0.png",
    "o3d prediction": "../../output/prediction_0.png",
    "p3d prediction": "../../output/prediction_p3d_0.png",
}
masks = {name: _foreground_mask(p) for name, p in paths.items()}
areas = {name: int(m.sum()) for name, m in masks.items()}
target_area = areas["target"]
print(f"{'image':<16} {'fg pixels':>10} {'area ratio':>12} {'linear ratio':>14}")
for name, area in areas.items():
    area_ratio = area / target_area
    linear_ratio = math.sqrt(area_ratio)
    print(f"{name:<16} {area:>10d} {area_ratio:>12.4f} {linear_ratio:>14.4f}")

# %%
# Bounding box comparison: separates uniform scale-shrink (bbox shrinks)
# from detail loss (bbox same, area smaller).
def _foreground_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())

target_r0, target_r1, target_c0, target_c1 = _foreground_bbox(masks["target"])
target_h = target_r1 - target_r0 + 1
target_w = target_c1 - target_c0 + 1
print(f"{'image':<16} {'bbox (r0,r1,c0,c1)':>22} {'h':>5} {'w':>5} {'h ratio':>9} {'w ratio':>9}")
for name, mask in masks.items():
    r0, r1, c0, c1 = _foreground_bbox(mask)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    print(f"{name:<16} {f'({r0},{r1},{c0},{c1})':>22} {h:>5d} {w:>5d} {h/target_h:>9.4f} {w/target_w:>9.4f}")
# %%
