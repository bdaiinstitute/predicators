"""Analysis for spot cube placing with active sampler learning."""


import os
from typing import Any, List, Optional, Tuple

import dill as pkl
import glob
import imageio
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize

from predicators import utils
from predicators.envs import BaseEnv, create_new_env
from predicators.envs.cover import BumpyCoverEnv
from predicators.ml_models import BinaryClassifier
from predicators.settings import CFG
from predicators.structs import EnvironmentTask, Object, State, Video, Image, Array


def _main() -> None:
    """Loads the saved samplers."""
    # Parse & validate args
    args = utils.parse_args()
    utils.update_config(args)
    _analyze_saved_data()


def _analyze_saved_data():
    """Use this to analyze the data saved in saved_datasets/."""
    nsrt_name = "PlaceToolNotHigh"
    objects_tuple_str = "spot:robot, cube:tool, extra_room_table:flat_surface"
    filepath_template = f"{CFG.data_dir}/{CFG.env}_{nsrt_name}({objects_tuple_str})_*.data"
    all_saved_files = glob.glob(filepath_template)
    assert all_saved_files
    regex = f"{CFG.data_dir}\/{CFG.env}_{nsrt_name}\({objects_tuple_str}\)_(\\d+).data"
    regex_matches = [re.match(regex, f) for f in all_saved_files]
    datapoint_ids = [int(m.groups()[0]) for m in regex_matches]
    print(f"Found {len(datapoint_ids)} data points")
    X, y = [], []
    for datapoint_id in datapoint_ids:
        filepath = filepath_template.replace("*", str(datapoint_id))
        with open(filepath, "rb") as f:
            X_i, y_i = pkl.load(f)
        X.extend(X_i)
        y.extend(y_i)
    img = _create_image(np.array(X), np.array(y))
    img_outfile = f"videos/spot_cube_active_sampler_learning_saved_data.png"
    imageio.imsave(img_outfile, img)
    print(f"Wrote out to {img_outfile}")


def _analyze_online_learning_cycles():
    """Use this to analyze the datasets saved after each cycle."""
    # Set up videos.
    video_frames = []
    # Evaluate samplers for each learning cycle.
    online_learning_cycle = 0
    while True:
        try:
            img = _run_one_cycle_analysis(online_learning_cycle)
            video_frames.append(img)
        except FileNotFoundError:
            break
        online_learning_cycle += 1
    # Save the video.
    video_outfile = f"spot_cube_active_sampler_learning.mp4"
    utils.save_video(video_outfile, video_frames)
    # Save the frames individually too.
    for t, img in enumerate(video_frames):
        img_outfile = f"videos/spot_cube_active_sampler_learning_{t}.png"
        imageio.imsave(img_outfile, img)


def _run_one_cycle_analysis(online_learning_cycle: Optional[int]) -> Image:
    option_name = "PlaceToolNotHigh"
    approach_save_path = utils.get_approach_save_path_str()
    save_path = f"{approach_save_path}_{option_name}_" + \
        f"{online_learning_cycle}.sampler_classifier"
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"File does not exist: {save_path}")
    with open(save_path, "rb") as f:
        classifier = pkl.load(f)
    print(f"Loaded sampler classifier from {save_path}.")
    save_path = f"{approach_save_path}_{option_name}_" + \
        f"{online_learning_cycle}.sampler_classifier_data"
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"File does not exist: {save_path}")
    with open(save_path, "rb") as f:
        training_data = pkl.load(f)
    print(f"Loaded sampler classifier training data from {save_path}.")
    X, y = training_data
    return _create_image(X, y, classifier=classifier)


def _create_image(X: List[Array], y: List[Array], classifier: Optional[BinaryClassifier] = None) -> Image:
    cmap = colormaps.get_cmap('RdYlGn')
    norm = Normalize(vmin=0.0, vmax=1.0)

    # x is [1.0, spot, tool, surface, params]
    # spot: "gripper_open_percentage", "curr_held_item_id", "x", "y", "z"
    # tool: "x", "y", "z", "lost", "in_view"
    # surface: "x", "y", "z"
    # params: "dx", "dy", "dz"
    assert X.shape[1] == 1 + 5 + 5 + 3 + 3

    fig, ax = plt.subplots(1, 1)

    x_min = 0
    x_max = 4
    y_min = -2
    y_max = 2
    density = 25
    radius = 0.05

    if classifier is not None:
        candidates = [(x, y) for x in np.linspace(x_min, x_max, density) for y in np.linspace(y_min, y_max, density)]
        for candidate in candidates:
            # Average scores over other possible values...?
            scores = []
            for standard_x in X:
                cand_x = standard_x.copy()
                cand_x[-3:-1] = candidate
                score = classifier.predict_proba(cand_x)
                scores.append(score)
            score = np.mean(scores)
            color = cmap(norm(score))
            circle = plt.Circle(candidate,
                                radius,
                                color=color,
                                alpha=0.1)
            ax.add_patch(circle)

    # plot real data
    for x, label in zip(X, y):
        x_param, y_param = x[-3:-1]
        color = cmap(norm(label))
        circle = plt.Circle((x_param, y_param),
                            radius,
                            color=color,
                            alpha=0.5)
        ax.add_patch(circle)
    
    plt.xlabel("x parameter")
    plt.ylabel("y parameter")
    plt.xlim((x_min - 3 * radius, x_max + 3 * radius))
    plt.ylim((y_min - 3 * radius, y_max + 3 * radius))

    return utils.fig2data(fig, dpi=150)


if __name__ == "__main__":
    _main()
