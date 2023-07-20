"""Analysis for kitchen active sampler learning."""

import glob
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import dill as pkl
import imageio
import matplotlib.pyplot as plt
import numpy as np
from bosdyn.client import math_helpers
from matplotlib import colormaps
from matplotlib.colors import Normalize

from predicators import utils
from predicators.ml_models import BinaryClassifier, MLPBinaryClassifier
from predicators.settings import CFG
from predicators.structs import Array, Image


def _main() -> None:
    # Parse & validate args
    args = utils.parse_args()
    utils.update_config(args)
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    logging.basicConfig(level=CFG.loglevel,
                        format="%(message)s",
                        handlers=handlers)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    _analyze_saved_data()


def _analyze_saved_data() -> None:
    """Use this to analyze the data saved in saved_datasets/."""
    nsrt_name = "MoveTo"
    objects_tuple_str = "gripper:gripper, kettle:obj"
    prefix = f"{CFG.data_dir}/{CFG.env}_{nsrt_name}({objects_tuple_str})_"
    filepath_template = f"{prefix}*.data"
    all_saved_files = glob.glob(filepath_template)
    X: List[Array] = []
    y: List[Array] = []
    times: List[int] = []
    for filepath in all_saved_files:
        with open(filepath, "rb") as f:
            datum = pkl.load(f)
        X_i, y_i = datum["datapoint"]
        time_i = datum["time"]
        X.append(X_i)
        y.append(y_i)
        times.append(time_i)
    idxs = [i for (i, _) in sorted(enumerate(times), key=lambda i: i[1])]
    X = [X[i] for i in idxs]
    y = [y[i] for i in idxs]
    img = _create_image(X, y)
    img_outfile = "kitchen_data_analysis.png"
    imageio.imsave(img_outfile, img)
    print(f"Wrote out to {img_outfile}")

def _create_image(X: List[Array],
                  y: List[Array]) -> Image:
    cmap = colormaps.get_cmap('RdYlGn')
    norm = Normalize(vmin=0.0, vmax=1.0)

    dims = [
        "constant", 
        "gripper x",
        "gripper y",
        "gripper z",
        "kettle x", 
        "kettle y",
        "kettle z",
        "kettle angle",
        "act dx",
        "act dy",
        "act dz"
    ]
    assert np.array(X).shape[1] == len(dims)

    fig, axes = plt.subplots(4, 3, sharex=False, sharey=False)

    rng = np.random.default_rng(0)

    # plot real data
    for i, (ax, dim) in enumerate(zip(axes.flat, dims)):
        min_x = np.inf
        max_x = -np.inf
        for datum, label in zip(X, y):
            x_pt = datum[i]
            min_x = min(min_x, x_pt)
            max_x = max(max_x, x_pt)
            color = cmap(norm(label))
            circle = plt.Circle((x_pt, label + rng.uniform(-0.1, 0.1)), 0.01, color=color, alpha=0.5)
            ax.add_patch(circle)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim((-0.1 + min_x, 0.1 + max_x))
        ax.set_title(dim)

    plt.tight_layout()

    return utils.fig2data(fig, dpi=150)


if __name__ == "__main__":
    _main()
