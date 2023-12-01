"""Analyze learned samplers for spot picking."""

import os
from typing import Any, List, Optional, Tuple

import dill as pkl
import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize

from predicators import utils
from predicators.envs import BaseEnv, create_new_env
from predicators.envs.spot_env import SpotRearrangementEnv, _movable_object_type
from predicators.settings import CFG
from predicators.structs import EnvironmentTask, Object, State, Video


def _main() -> None:
    """Loads the saved samplers."""
    # Parse & validate args
    args = utils.parse_args()
    utils.update_config(args)
    env = create_new_env(CFG.env, do_cache=True)
    # Create separate plots for each movable object in the environment.
    assert isinstance(env, SpotRearrangementEnv)
    graspable_objs = {o for o in env._detection_id_to_obj.values() if o.is_instance(_movable_object_type)}
    for obj in graspable_objs:
        print(f"Starting analysis for {obj.name}")
        # Set up videos.
        video_frames = []
        # Evaluate samplers before offline learning.
        imgs = _run_one_cycle_analysis(None, obj, env)
        video_frames.append(imgs)
        # Evaluate samplers for each learning cycle.
        online_learning_cycle = 1
        while True:
            try:
                imgs = _run_one_cycle_analysis(online_learning_cycle, obj,
                                            env)
                video_frames.append(imgs)
            except FileNotFoundError:
                break
            online_learning_cycle += 1
        # Save the videos.
        for i, video in enumerate(np.swapaxes(video_frames, 0, 1)):
            video_outfile = f"spot_pick_sampler_learning_case_{i}.mp4"
            utils.save_video(video_outfile, video)



def _run_one_cycle_analysis(online_learning_cycle: Optional[int],
                            target_object: Object,
                            env: BaseEnv) -> Video:
    import ipdb; ipdb.set_trace()

    return imgs


if __name__ == "__main__":
    _main()

