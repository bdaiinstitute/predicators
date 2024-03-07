import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from predicators import utils
from predicators.envs.pddl_env import _parse_pddl_domain
import imageio.v2 as iio
import cv2
import pickle as p


def _oven_open_classifier(state):
    # Count the number of active voxels in the expected region and threshold.
    x_min = 8
    x_max = 32
    z_min = 8
    z_max = 64 - 8
    y_min = 64 - 64 // 5
    y_max = 63
    region = state[x_min:x_max, y_min:y_max, z_min:z_max]
    num_active = np.sum(np.array(region[..., 3].flat) > 1e-5)
    return num_active > 100


def _oven_closed_classifier(state):
    return not _oven_open_classifier(state)



_PREDICATE_CLASSIFIERS = {
    # TODO: notholdingbagel
    # TODO: bagelgrasped
    # TODO: bagelontable
    # TODO: bagelontray
    # TODO: trayinsideoven
    # TODO: traypulledout
    "ovenopen": _oven_open_classifier,
    "ovenclosed": _oven_closed_classifier,
}


def _canonicalize_voxel_map(voxel_map):
    # Assume that the y dimension is consistent. Crop above a certain y value
    # to isolate the top of the oven, which we will use to orient the scene.
    cropped_voxel_map = voxel_map[:, 10:25]
    oven_xs, oven_zs, _ = collapse_voxel_map(cropped_voxel_map, 0, 2, "forward")

    # Remove outliers.
    mask_image = np.zeros((voxel_map.shape[0], voxel_map.shape[2]), dtype=np.uint8)
    for x, z in zip(oven_xs, oven_zs):
        mask_image[x, z] = 1
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(mask_image, connectivity=8)
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    points = np.argwhere(output == max_label)

    # Fit a rectangle to the points.
    (cx, cz), (w, h), rot_deg  = cv2.minAreaRect(points)

    # Uncomment to debug.
    # rect = utils.Rectangle.from_center(cx, cz, w, h, rotation_about_center=rot_deg / 180 * np.pi)
    # _, ax = plt.subplots(1, 1)
    # ax.scatter(oven_xs, oven_zs)
    # rect.plot(ax, facecolor=(1, 1, 1, 0.5), edgecolor="black")
    # ax.set_xlim([0, voxel_map.shape[0]])
    # ax.set_ylim([0, voxel_map.shape[2]])
    # ax.set_xlabel("x")
    # ax.set_ylabel("z")
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.savefig("debug.png")

    # Determine the angle between the oven and the center of the workspace.
    wx = (voxel_map.shape[0] - 1) / 2
    wz = (voxel_map.shape[2] - 1) / 2
    rot = np.pi - np.arctan2(cz - wz, cx - wx)  # between -pi and pi
    rot_degrees = 180 * rot / np.pi

    # Rotate points in the voxel map by that amount.
    new_voxel_map = np.zeros_like(voxel_map)
    for x in range(voxel_map.shape[0]):
        for z in range(voxel_map.shape[2]):
            new_x, new_z = utils.rotate_point_in_image(x, z, rot_degrees, voxel_map.shape[0],
                            voxel_map.shape[2])
            for y in range(voxel_map.shape[1]):
                new_voxel_map[new_x, y, new_z] = voxel_map[x, y, z]

    # Uncomment to debug.
    # img = voxel_map_to_img(new_voxel_map)
    # global COUNT
    # COUNT += 1
    # iio.imsave(f"debug_imgs/debug_rot_{COUNT}.png", img)
    # raise NotImplementedError

    return new_voxel_map


def _get_state_from_voxel_map(voxel_map):
    return _canonicalize_voxel_map(voxel_map)


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
    img = utils.fig2data(fig, dpi=150)
    plt.close()
    return img


def load_data(demo_num):
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

    return voxels



def create_voxel_map_video(demo_num):

    voxels = load_data(demo_num)

    dirpath =  Path("/Users/tom/Desktop") / "equidiff"
    annotations_filepath = dirpath  / f"bagel_oven_annotations_demo{demo_num}.p"
    annotations = None
    if annotations_filepath.exists():
        with open(annotations_filepath, "rb") as f:
            annotations = p.load(f)

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
    voxels = load_data(demo_num)

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



def _test_oven_open_closed_classifier():
     voxels = load_data(demo_num=0)
     neg1 = np.swapaxes(voxels[0], 0, -1)
     assert not _oven_open_classifier(neg1)
     assert _oven_closed_classifier(neg1)

     pos1 = np.swapaxes(voxels[200], 0, -1)
     assert _oven_open_classifier(pos1)
     assert not _oven_closed_classifier(pos1)


if __name__ == "__main__":
    create_voxel_map_video(demo_num=0)
    # create_predicate_annotations(demo_num=40)

    # _test_oven_open_closed_classifier()