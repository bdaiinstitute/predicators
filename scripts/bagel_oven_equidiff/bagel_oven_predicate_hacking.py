import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from predicators import utils
from predicators.envs.pddl_env import _parse_pddl_domain
import imageio.v2 as iio
import cv2
import pickle as p


def _tray_pulled_out_classifier(state):
    return False  # TODO


def _tray_inside_oven_classifier(state):
    return not _tray_pulled_out_classifier(state)



_PREDICATE_CLASSIFIERS = {
    # TODO: notholdingbagel
    # TODO: ovenclosed
    # TODO: ovenopen
    # TODO: bagelgrasped
    # TODO: bagelontable
    # TODO: bagelontray
    "trayinsideoven": _tray_inside_oven_classifier,
    "traypulledout": _tray_pulled_out_classifier,
}


def _get_state_from_voxel_map(voxel_map):
    # Canonicalize orientation with respect to the oven.
    return voxel_map


def get_abstract_state(voxel_map):
    state = _get_state_from_voxel_map(voxel_map)
    curr_pred_names = set()
    for pred_name, classifier in _PREDICATE_CLASSIFIERS.items():
        if classifier(state):
            curr_pred_names.add(pred_name)
    return sorted(curr_pred_names)





def collapse_voxel_map(voxel_map, dim0, dim1, dim2_direction):
    dim2s = [i for i in range(3) if i not in {dim0, dim1}]
    assert len(dim2s) == 1
    dim2 = dim2s[0]
    x_coords, y_coords, colors = [], [], []
    for d0 in range(voxel_map.shape[dim0]):
        for d1 in range(voxel_map.shape[dim1]):
            if dim2_direction == "forward":
                d2s = range(voxel_map.shape[dim2])
            else:
                assert dim2_direction == "backward"
                d2s = range(voxel_map.shape[dim2] - 1, -1, -1)
            for d2 in d2s:
                d_to_i = {dim0: d0, dim1: d1, dim2: d2}
                idx = (d_to_i[0], d_to_i[1], d_to_i[2])
                if voxel_map[idx + (3,)] > 0:
                    x_coords.append(d0)
                    y_coords.append(d1)
                    colors.append(voxel_map[idx] / 255)
                    break
    return x_coords, y_coords, colors



def voxel_map_to_img(voxel_map, title=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # TODO remove
    voxel_map_shape = voxel_map.shape
    # voxel_map = voxel_map[:, :25, :]  # CROPS JUST THE TOP OF THE VOXELS

    # X/Y
    ax = axes[0]
    x_coords, y_coords, colors = collapse_voxel_map(voxel_map, 0, 1, "backward")
    ax.scatter(x_coords, y_coords, c=colors)
    ax.set_xlim([0, voxel_map_shape[0]])
    ax.set_ylim([voxel_map_shape[1], 0])
    ax.set_xlabel("x")
    ax.set_ylabel("-y")
    # ax.set_xticks([])
    # ax.set_yticks([])

    # X/Z
    ax = axes[1]
    x_coords, z_coords, colors = collapse_voxel_map(voxel_map, 0, 2, "forward")
    ax.scatter(x_coords, z_coords, c=colors)
    ax.set_xlim([0, voxel_map_shape[0]])
    ax.set_ylim([0, voxel_map_shape[2]])
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    # ax.set_xticks([])
    # ax.set_yticks([])

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
        title = ""
        if annotations and len(annotations) >= t-1:
            annotations_t = annotations[t]
            title = f"Annotations: {annotations_t}"

        predicted_pred_names = get_abstract_state(voxel_map)
        title += f"\nPredictions: {predicted_pred_names}"

        img = voxel_map_to_img(voxel_map, title=title)
        imgs.append(img)

        # TODO remove
        if t > 5:
            break

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
    create_voxel_map_video(demo_num=40)
    # create_predicate_annotations(demo_num=40)
