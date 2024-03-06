import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from predicators import utils
import imageio.v2 as iio

dirpath =  Path("/Users/tom/Desktop") / "equidiff"
filepath = dirpath / "data_teleop_oven_full_x58.hdf5"
assert filepath.exists()

f = h5py.File(filepath, 'r')
dataset = f['dataset']
demo = dataset['demo_0']
actions = demo["action"]
voxels = demo["agentview_voxel"]
eef_pos = demo["robot0_eef_pos"]
eef_quat = demo["robot0_eef_quat"]
gripper_qpos = demo["robot0_gripper_qpos"]

imgs = []
for t in range(len(voxels)):
    voxel_map = np.swapaxes(voxels[t], 0, -1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # X/Y
    ax = axes[0]
    x_coords, y_coords, colors = [], [], []
    for x in range(voxel_map.shape[0]):
        for y in range(voxel_map.shape[1]):
            for z in range(voxel_map.shape[2]):
                if voxel_map[x, y, z, 3] > 0:
                    x_coords.append(x)
                    y_coords.append(y)
                    colors.append(voxel_map[x, y, z] / 255)
                    break
    ax.scatter(x_coords, y_coords, c=colors)
    ax.set_xlim([0, voxel_map.shape[0]])
    ax.set_ylim([0, voxel_map.shape[1]])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks([])
    ax.set_yticks([])

    # X/Z
    ax = axes[1]
    x_coords, z_coords, colors = [], [], []
    for x in range(voxel_map.shape[0]):
        for z in range(voxel_map.shape[2]):
            for y in range(voxel_map.shape[1]):
                if voxel_map[x, y, z, 3] > 0:
                    x_coords.append(x)
                    z_coords.append(z)
                    colors.append(voxel_map[x, y, z] / 255)
                    break
    ax.scatter(x_coords, z_coords, c=colors)
    ax.set_xlim([0, voxel_map.shape[0]])
    ax.set_ylim([0, voxel_map.shape[2]])
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_xticks([])
    ax.set_yticks([])

    # Show the plot
    img = utils.fig2img(fig, dpi=150)
    plt.close()
    imgs.append(img)

video_path = dirpath / "bagel_oven_viz.mp4"
iio.mimsave(video_path, imgs, fps=10)
print(f"Wrote out to {video_path}")


#### From ChatGPT ####

# # Assuming `voxel_map` is your (4, 64, 64, 64) array
# # Rearrange the array to (64, 64, 64, 4) for RGBA to be the last dimension
# voxel_map = np.transpose(voxel_map, (1, 2, 3, 0))

# # Create a figure and 3D axis
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Prepare arrays to hold voxel coordinates and colors
# x_coords, y_coords, z_coords, colors = [], [], [], []

# # Iterate over each voxel
# for x in range(voxel_map.shape[0]):
#     for y in range(voxel_map.shape[1]):
#         for z in range(voxel_map.shape[2]):
#             # Check if the voxel is not completely transparent (alpha channel is not 0)
#             if voxel_map[x, y, z, 3] > 0:
#                 x_coords.append(x)
#                 y_coords.append(y)
#                 z_coords.append(z)
#                 colors.append(voxel_map[x, y, z] / 255)  # Extract RGB, ignore Alpha

# # Scatter plot (use 'c' for color)
# ax.scatter(x_coords, y_coords, z_coords, c=colors)

# # Set plot limits
# ax.set_xlim([0, voxel_map.shape[0]])
# ax.set_ylim([0, voxel_map.shape[1]])
# ax.set_zlim([0, voxel_map.shape[2]])

# # Show the plot
# plt.show()
