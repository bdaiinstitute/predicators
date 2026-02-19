import numpy as np
import json
from predicators.envs.spot_env import LISSpotCollectEnv
from predicators.perception.spot_perceiver import SpotPerceiver
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.structs import Object, State, GroundAtom
from predicators.spot_utils.utils import _container_type, _movable_object_type, _robot_type
from predicators import utils

utils.reset_config({
    "env":
    "lis_spot_collect_misplaced_items_env",
    "approach":
    "spot_wrapper[oracle]",
    "num_train_tasks":
    0,
    "num_test_tasks":
    1,
    "seed":
    0,
    "spot_run_dry":
    True,
    "spot_robot_ip":
    None,
    "spot_graph_nav_map":
    "b45-621",
    "bilevel_plan_without_sim":
    True,
    "perceiver":
    "spot_perceiver",
    "spot_use_perfect_samplers":
    True,
})

rng = np.random.default_rng(123)
env = LISSpotCollectEnv()
perceiver = SpotPerceiver()
nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                        get_gt_options(env.get_name()))

pred_name_to_pred = {p.name: p for p in env.predicates}

robot = Object("robot", _robot_type)
handle = Object("green_handle", _movable_object_type)
blue_block = Object("blue_block", _movable_object_type)
yellow_cup = Object("yellow_cup", _movable_object_type)
toy_plane = Object("toy_plane", _movable_object_type)
cardboard_box = Object("cardboard_box", _container_type)
Inside = pred_name_to_pred["Inside"]

# state from json
state = State({}, simulator_state=None)

# get all grounded atoms
all_grounded_atoms = None

json_file = "/Users/shashlik/Desktop/last.json"
with open(json_file, "r", encoding="utf-8") as f:
    json_dict = json.load(f)
object_name_to_object = env._parse_object_name_to_object_from_json(
    json_dict)
init_dict = env._parse_init_state_dict_from_json(
    json_dict, object_name_to_object)
for i, (obj, init_val) in enumerate(sorted(init_dict.items())):
            init_val["object_id"] = i
state = utils.create_state_from_dict(init_dict)

a = GroundAtom(Inside, [blue_block, cardboard_box])
print(a, a.holds(state))

import ipdb; ipdb.set_trace()

for a in all_grounded_atoms:
    a.holds(state)