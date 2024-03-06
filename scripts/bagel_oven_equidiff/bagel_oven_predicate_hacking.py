import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from predicators import utils
from predicators.envs.pddl_env import _parse_pddl_domain
import imageio.v2 as iio
import cv2
import pickle as p


def voxel_map_to_img(voxel_map, title=None):
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
    if title:
        plt.suptitle(title)

    # Show the plot
    img = utils.fig2data(fig, dpi=150)
    plt.close()
    return img


def create_voxel_map_video(demo_num):

    dirpath =  Path("/Users/tom/Desktop") / "equidiff"
    filepath = dirpath / "data_teleop_oven_full_x58.hdf5"
    assert filepath.exists()

    annotations_filepath = dirpath  / f"bagel_oven_annotations_demo{demo_num}.p"
    annotations = None
    if annotations_filepath.exists():
        with open(annotations_filepath, "rb") as f:
            annotations = p.load(f)

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
        title = None
        if annotations and len(annotations) >= t-1:
            annotations_t = annotations[t]
            title = f"Annotations: {annotations_t}"
        img = voxel_map_to_img(voxel_map, title=title)
        imgs.append(img)

    video_path = dirpath / f"bagel_oven_viz_demo{demo_num}.mp4"
    iio.mimsave(video_path, imgs, fps=10)
    print(f"Wrote out to {video_path}")


def create_predicate_annotations(demo_num):

    # Load predicates from domain.pddl in same directory
    domain_filepath = Path(__file__).parent / "domain.pddl"
    with open(domain_filepath, "r") as f:
        domain_str = f.read()
    _, predicates, _ = _parse_pddl_domain(domain_str)
    sorted_pred_names = sorted(p.name for p in predicates)

    dirpath =  Path("/Users/tom/Desktop") / "equidiff"
    filepath = dirpath / "data_teleop_oven_full_x58.hdf5"
    assert filepath.exists()

    f = h5py.File(filepath, 'r')
    dataset = f['dataset']
    demo = dataset[f'demo_{demo_num}']
    voxels = demo["agentview_voxel"]

    annotations = []
    pred_prompt_str = ", ".join(f"{i}: {p}" for i, p in enumerate(sorted_pred_names))
    prompt = f"Which of the following predicates hold? Enter a comma-separated list of integers. Enter 's' for same as last. {pred_prompt_str}\n"

    for t in range(len(voxels)):
        voxel_map = np.swapaxes(voxels[t], 0, -1)
        rgb_img = voxel_map_to_img(voxel_map, title=f"Time Step {t}")

        cv2.imshow('img', cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

        while True:
            res = input(prompt)
            
            try:
                if res == 's':
                    annotation = annotations[-1]
                else:
                    pred_names = set()
                    for i_str in res.split(","):
                        i_str = i_str.strip()
                        assert i_str.isdigit()
                        i = int(i_str)
                        assert 0 <= i < len(sorted_pred_names)
                        pred_name = sorted_pred_names[i]
                        pred_names.add(pred_name)
                    annotation = sorted(pred_names)
            except:
                pass
            
            annotations.append(annotation)
            break

    annotations_path = dirpath / f"bagel_oven_annotations_demo{demo_num}.p"
    with open(annotations_path, "wb") as f:
        p.dump(annotations, f)
    print(f"Dumped annotations to {annotations_path}")


if __name__ == "__main__":
    create_voxel_map_video(demo_num=0)
    # create_predicate_annotations(demo_num=0)