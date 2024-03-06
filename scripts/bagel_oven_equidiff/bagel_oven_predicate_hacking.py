import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from predicators import utils
import imageio.v2 as iio

def voxel_map_to_img(voxel_map):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # X/Y
    ax = axes[0]
    x_coords, y_coords, colors = [], [], []
    for x in range(voxel_map.shape[0]):
        for y in range(voxel_map.shape[1]):
            for z in range(voxel_map.shape[2] - 1, -1, -1):  # NOTE
                if voxel_map[x, y, z, 3] > 0:
                    x_coords.append(x)
                    y_coords.append(y)
                    colors.append(voxel_map[x, y, z] / 255)
                    break
    ax.scatter(x_coords, y_coords, c=colors)
    ax.set_xlim([0, voxel_map.shape[0]])
    ax.set_ylim([voxel_map.shape[1], 0])
    ax.set_xlabel("x")
    ax.set_ylabel("-y")
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
    img = utils.fig2data(fig, dpi=150)
    plt.close()
    return img


def create_voxel_map_video(demo_num):

    dirpath =  Path("/Users/tom/Desktop") / "equidiff"
    filepath = dirpath / "data_teleop_oven_full_x58.hdf5"
    assert filepath.exists()

    f = h5py.File(filepath, 'r')
    dataset = f['dataset']
    demo = dataset[f'demo_{demo_num}']
    voxels = demo["agentview_voxel"]

    # TODO: use?
    eef_pos = demo["robot0_eef_pos"]
    eef_quat = demo["robot0_eef_quat"]
    gripper_qpos = demo["robot0_gripper_qpos"]

    imgs = []

    for t in range(len(voxels)):
        voxel_map = np.swapaxes(voxels[t], 0, -1)
        img = voxel_map_to_img(voxel_map)
        imgs.append(img)

    video_path = dirpath / f"bagel_oven_viz_demo{demo_num}.mp4"
    iio.mimsave(video_path, imgs, fps=10)
    print(f"Wrote out to {video_path}")

if __name__ == "__main__":
    create_voxel_map_video(demo_num=40)