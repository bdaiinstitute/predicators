import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

filepath = Path("/Users/tom/Desktop") / "equidiff"/ "data_teleop_oven_full_x58.hdf5"
assert filepath.exists()

f = h5py.File(filepath, 'r')
dataset = f['dataset']
demo = dataset['demo_0']
actions = demo["action"]
voxels = demo["agentview_voxel"]
eef_pos = demo["robot0_eef_pos"]
eef_quat = demo["robot0_eef_quat"]
gripper_qpos = demo["robot0_gripper_qpos"]

voxel_map = voxels[0]

#### From ChatGPT ####

# Assuming `voxel_map` is your (4, 64, 64, 64) array
# Rearrange the array to (64, 64, 64, 4) for RGBA to be the last dimension
voxel_map = np.transpose(voxel_map, (1, 2, 3, 0))

# Create a figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Prepare arrays to hold voxel coordinates and colors
x_coords, y_coords, z_coords, colors = [], [], [], []

# Iterate over each voxel
for x in range(voxel_map.shape[0]):
    for y in range(voxel_map.shape[1]):
        for z in range(voxel_map.shape[2]):
            # Check if the voxel is not completely transparent (alpha channel is not 0)
            if voxel_map[x, y, z, 3] > 0:
                x_coords.append(x)
                y_coords.append(y)
                z_coords.append(z)
                colors.append(voxel_map[x, y, z] / 255)  # Extract RGB, ignore Alpha

# Scatter plot (use 'c' for color)
ax.scatter(x_coords, y_coords, z_coords, c=colors)

# Set plot limits
ax.set_xlim([0, voxel_map.shape[0]])
ax.set_ylim([0, voxel_map.shape[1]])
ax.set_zlim([0, voxel_map.shape[2]])

# Show the plot
plt.show()