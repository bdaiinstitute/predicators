"""Functions to help visualize learned samplers."""

from typing import List
import dill as pkl
import matplotlib.pyplot as plt
import os
import numpy as np
import imageio

from matplotlib import colormaps
from matplotlib.colors import Normalize

from predicators.settings import CFG
from predicators import utils
from predicators.spot_utils.utils import load_spot_metadata
from predicators.structs import Video


def _main():
    """Loads the saved samplers."""
    # Parse & validate args
    args = utils.parse_args()
    utils.update_config(args)
    cycles_to_plot = [11]
    imgs = visualize_cup_table_place_samplers(cycles_to_plot, 25)
    for t, img in zip(cycles_to_plot, imgs):
        img_outfile = f"videos/cup_table_active_sampler_learning_cycle_{t}.png"
        imageio.imsave(img_outfile, img)
        print(f"Saved sampler analysis figure {img_outfile}.")


def visualize_cup_table_place_samplers(online_learning_cycles: List, num_samples_len_and_wid: int) -> Video:
    """Create a visualization of the learned place cup on table sampler
    at different online learning cycles."""
    assert CFG.env == "spot_ball_and_cup_sticky_table_env"
    assert CFG.active_sampler_learning_feature_selection == "oracle"
    cmap = colormaps.get_cmap('RdYlGn')
    norm = Normalize(vmin=0.0, vmax=1.0)
    drafting_table_feats = load_spot_metadata()["static-object-features"]["drafting_table"]
    drafting_table_len = drafting_table_feats["length"]
    drafting_table_wid = drafting_table_feats["width"]
    sticky_region_x = drafting_table_feats["sticky-region-x"]
    sticky_region_y = drafting_table_feats["sticky-region-y"]
    len_vals = np.linspace(-drafting_table_len/2, drafting_table_len/2, num_samples_len_and_wid)
    wid_vals = np.linspace(-drafting_table_wid/2, drafting_table_wid/2, num_samples_len_and_wid)
    imgs = []
    # Produce a new image for every online learning cycle.
    for online_learning_cycle in online_learning_cycles:
        fig, ax = plt.subplots(1, 1)
        table_geom = utils.Rectangle.from_center(0.0, 0.0, drafting_table_wid, drafting_table_len, 0.0)
        table_geom.plot(ax, **{'color': 'white'})
        option_name = "PlaceObjectOnTop"
        approach_save_path = utils.get_approach_save_path_str()
        cls_save_path = f"{approach_save_path}_{option_name}_" + \
            f"{online_learning_cycle}.sampler_classifier"
        cls_data_save_path = f"{approach_save_path}_{option_name}_" + \
            f"{online_learning_cycle}.sampler_classifier_data"
        if not os.path.exists(cls_save_path) or not os.path.exists(cls_data_save_path):
            raise FileNotFoundError
        with open(cls_save_path, "rb") as f:
            classifier = pkl.load(f)
        with open(cls_data_save_path, "rb") as f:
            classifier_data = pkl.load(f)
        print(f"Loaded sampler classifier and data from {cls_save_path}.")
        assert len(classifier_data[0][0]) == 7
        table_id = classifier_data[0][0][1]
        drop_height = 0.0 #classifier_data[0][0][-1]
        for y in len_vals:
            for x in wid_vals:
                sampler_input = [1.0, table_id, sticky_region_x, sticky_region_y, x, y, drop_height]
                score = classifier.predict_proba(np.array(sampler_input))
                import ipdb; ipdb.set_trace()
                color = cmap(norm(score))
                circle = plt.Circle((x, y),
                                    0.005,
                                    color=color,
                                    alpha=0.1)
                ax.add_patch(circle)
        ax.set_xlim(-drafting_table_wid/2 - 0.1, drafting_table_wid/2 + 0.1)
        ax.set_ylim(-drafting_table_len/2 - 0.1, drafting_table_len/2 + 0.1)
        img = utils.fig2data(fig, dpi=150)
        imgs.append(img)

    return imgs

if __name__ == "__main__":
    _main()

