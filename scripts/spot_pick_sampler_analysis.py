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
from predicators.envs import create_new_env
from predicators.envs.spot_env import SpotRearrangementEnv, \
    _movable_object_type, _robot_type
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.perception.spot_perceiver import SpotPerceiver
from predicators.settings import CFG
from predicators.structs import Array, EnvironmentTask, NSRTSampler, Object, \
    ParameterizedOption, State, Video


def _main() -> None:
    """Loads the saved samplers."""
    # Parse & validate args
    args = utils.parse_args()
    utils.update_config(args)
    env = create_new_env(CFG.env, do_cache=True)
    assert isinstance(env, SpotRearrangementEnv)
    rng = np.random.default_rng(CFG.seed)
    # Create an example state that includes the objects of interest. The actual
    # state should not be used.
    state = _create_example_state(env)
    robot, = state.get_objects(_robot_type)
    surface = state.get_objects(_movable_object_type)[0]  # shouldn't matter
    # Load the parameterized option of interest.
    skill_name = "PickObjectFromTop"
    options = get_gt_options(env.get_name())
    option = next(o for o in options if o.name == skill_name)
    # Load the base sampler.
    nsrts = get_gt_nsrts(env.get_name(), env.predicates, options)
    nsrt = next(n for n in nsrts if n.name == skill_name)
    base_sampler = nsrt._sampler
    # Create separate plots for each movable object in the environment.
    graspable_objs = {o for o in state if o.is_instance(_movable_object_type)}
    for obj in graspable_objs:
        print(f"Starting analysis for {obj.name}")
        # Create the inputs to the NSRT / option.
        object_inputs = [robot, obj, surface]
        # Create candidate samplers.
        num_candidates = 100
        candidates = [
            base_sampler(state, set(), rng, object_inputs)
            for _ in range(num_candidates)
        ]
        # Set up videos.
        video_frames = []
        # Evaluate samplers for each learning cycle.
        online_learning_cycle = 0
        while True:
            try:
                img = _run_one_cycle_analysis(online_learning_cycle, obj,
                                              state, option, object_inputs,
                                              candidates, rng)
                video_frames.append(img)
            except FileNotFoundError:
                break
            online_learning_cycle += 1
        # Save the videos.
        video_outfile = f"spot_pick_sampler_learning_{obj.name}.mp4"
        utils.save_video(video_outfile, video_frames)


def _run_one_cycle_analysis(online_learning_cycle: Optional[int],
                            target_object: Object, example_state: State,
                            param_option: ParameterizedOption,
                            object_inputs: List[Object],
                            candidates: List[Array],
                            rng: np.random.Generator) -> Video:
    option_name = param_option.name
    approach_save_path = utils.get_approach_save_path_str()
    save_path = f"{approach_save_path}_{option_name}_" + \
        f"{online_learning_cycle}.sampler_classifier"
    if not os.path.exists(save_path):
        raise FileNotFoundError
    with open(save_path, "rb") as f:
        classifier = pkl.load(f)
    print(f"Loaded sampler classifier from {save_path}.")

    cmap = colormaps.get_cmap('RdYlGn')
    norm = Normalize(vmin=0.0, vmax=1.0)

    # Classify the candidates.
    predictions = []
    for candidate in candidates:
        sampler_input = utils.construct_active_sampler_input(
            example_state, object_inputs, candidate, param_option)
        prediction = classifier.predict_proba(sampler_input)
        predictions.append(prediction)

    # Visualize the classifications.
    fig, ax = plt.subplots(1, 1)
    radius = 0.5
    for candidate, prediction in zip(candidates, predictions):
        x, y = candidate[:2]
        color = cmap(norm(prediction))
        circle = plt.Circle((x, y), radius, color=color, alpha=0.5)
        ax.add_patch(circle)

    plt.title(f"{target_object.name} Cycle {online_learning_cycle}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0, 100))
    plt.ylim((0, 100))

    return utils.fig2data(fig, dpi=150)


def _create_example_state(env: SpotRearrangementEnv) -> State:
    perceiver = SpotPerceiver()
    empty_task = env.get_task("train", 0)
    perceiver.reset(empty_task)
    init_obs = env.reset("train", 0)
    init_state = perceiver.step(init_obs)
    return init_state.copy()


if __name__ == "__main__":
    _main()
