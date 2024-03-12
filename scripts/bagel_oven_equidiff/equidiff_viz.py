import h5py
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as iio
from tqdm import tqdm

matplotlib.use("TkAgg")


def fig2data(fig, dpi=150):
    fig.set_dpi(dpi)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).copy()
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4, ))
    data[..., [0, 1, 2, 3]] = data[..., [1, 2, 3, 0]]
    return data


def collapse_voxel_map(voxel_map, dim0, dim1, dim2_direction):
    dim2s = [i for i in range(3) if i not in {dim0, dim1}]
    assert len(dim2s) == 1
    dim2 = dim2s[0]

    keep_mask = np.ones((voxel_map.shape[dim0], voxel_map.shape[dim1]), dtype=bool)

    if dim2_direction == "forward":
        d2s = range(voxel_map.shape[dim2])
    else:
        assert dim2_direction == "backward"
        d2s = range(voxel_map.shape[dim2] - 1, -1, -1)

    coords0, coords1, colors = [], [], []
    for d2 in d2s:
        d_to_i = {dim0: slice(None), dim1: slice(None), dim2: d2}
        idx = (d_to_i[0], d_to_i[1], d_to_i[2])
        voxel_map_layer = voxel_map[idx]
        this_keep_mask = keep_mask & (voxel_map_layer[..., 3] > 0)
        for coord0, coord1 in np.argwhere(this_keep_mask):
            # This is probably wrong!
            color = voxel_map_layer[(coord0, coord1)] / 255
            coords0.append(coord0)
            coords1.append(coord1)
            colors.append(color)
        keep_mask = keep_mask & (~this_keep_mask)

    return coords0, coords1, colors


def voxel_map_to_img(voxel_map, title=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    voxel_map_shape = voxel_map.shape

    # X/Y
    ax = axes[0]
    x_coords, y_coords, colors = collapse_voxel_map(voxel_map, 0, 1, "backward")
    ax.scatter(x_coords, y_coords, c=colors)
    ax.set_xlim([0, voxel_map_shape[0]])
    ax.set_ylim([voxel_map_shape[1], 0])
    ax.set_xlabel("x")
    ax.set_ylabel("-y")
    ax.set_xticks([])
    ax.set_yticks([])

    # X/Z
    ax = axes[1]
    x_coords, z_coords, colors = collapse_voxel_map(voxel_map, 0, 2, "forward")
    ax.scatter(x_coords, z_coords, c=colors)
    ax.set_xlim([0, voxel_map_shape[0]])
    ax.set_ylim([0, voxel_map_shape[2]])
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        plt.suptitle(title)

    # Show the plot
    img = fig2data(fig)
    plt.close()
    return img


def load_demo_voxels(path_to_hdf5, demo_num):
    f = h5py.File(path_to_hdf5, 'r')
    dataset = f['dataset']
    demo = dataset[f'demo_{demo_num}']
    voxels = np.array(demo["agentview_voxel"])
    f.close()
    return voxels


def create_voxel_map_video(voxels):
    imgs = []
    for t in tqdm(range(len(voxels))):
        voxel_map = np.swapaxes(voxels[t], 0, -1)
        title = f"\nTime {t} "
        img = voxel_map_to_img(voxel_map, title=title)
        imgs.append(img)
    return imgs


if __name__ == "__main__":
    dirpath = Path("/Users/tom/Dropbox") / "equidiff"
    path_to_hdf5 =  dirpath / "data_teleop_oven_full_x58.hdf5"
    demo_num = 5
    voxels = load_demo_voxels(path_to_hdf5, demo_num)
    imgs = create_voxel_map_video(voxels)
    videopath = dirpath / f"demo_viz{demo_num}.mp4"
    iio.mimsave(videopath, imgs, fps=30)
    print(f"Wrote out video to {videopath}")
