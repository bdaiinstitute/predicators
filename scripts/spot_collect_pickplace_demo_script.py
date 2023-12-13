import numpy as np
import pickle

from predicators import utils
from predicators.envs.spot_env import SpotCleanupShelfEnv
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.perception.spot_perceiver import SpotPerceiver
from predicators.settings import CFG
from predicators.spot_utils.skills.spot_navigation import go_home
from predicators.structs import Action, GroundAtom, _GroundNSRT


def pick_and_place_script() -> None:
    args = utils.parse_args(env_required=False,
                            seed_required=False,
                            approach_required=False)
    utils.reset_config({
        "spot_graph_nav_map": "movo_room",
        "env": "spot_cleanup_shelf_env",
        "approach": "spot_wrapper[oracle]",
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "seed": 123,
        "spot_robot_ip": args["spot_robot_ip"],
        "test_task_json_dir": args.get("test_task_json_dir", None),
    })
    print(CFG.spot_graph_nav_map)
    rng = np.random.default_rng(123)
    env = SpotCleanupShelfEnv()
    perceiver = SpotPerceiver()
    nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                         get_gt_options(env.get_name()))

    # Lists to store demo
    observation_list = []
    state_list = []
    action_list = []
    option_list = []

    # This should have the robot spin around and locate all objects.
    task = env.get_test_tasks()[0]
    obs = env.reset("test", 0)
    perceiver.reset(task)
    print(obs.objects_in_view)
    cabinet, cube, floor, shelf = sorted(obs.objects_in_view)
    assert cabinet.name == "cabinet"
    assert cube.name == "cube"
    assert floor.name == "floor"
    assert shelf.name == "shelf"

    state = perceiver.step(obs)
    spot = next(o for o in state if o.type.name == "robot")
    nsrt_name_to_nsrt = {n.name: n for n in nsrts}
    MoveToHandViewObject = nsrt_name_to_nsrt["MoveToHandViewObject"]
    MoveToReachObject = nsrt_name_to_nsrt["MoveToReachObject"]
    PickObjectFromSide = nsrt_name_to_nsrt["PickObjectFromSide"]
    PlaceObjectOnTop = nsrt_name_to_nsrt["PlaceObjectOnTop"]

    # The robot gripper should be empty, and the chair should be visible
    pred_name_to_pred = {p.name: p for p in env.predicates}
    HandEmpty = pred_name_to_pred["HandEmpty"]
    On = pred_name_to_pred["On"]
    InHandView = pred_name_to_pred["InHandView"]
    Holding = pred_name_to_pred["Holding"]
    Reachable = pred_name_to_pred["Reachable"]
    assert GroundAtom(HandEmpty, [spot]).holds(state)
    assert GroundAtom(On, [cube, shelf]).holds(state)

    # Sample and run an option to move to the cube
    move_to_cube_nsrt = MoveToHandViewObject.ground((spot, cube))
    assert all(a.holds(state) for a in move_to_cube_nsrt.preconditions), f"{[(a, a.holds(state)) for a in move_to_cube_nsrt.preconditions]}"
    option = move_to_cube_nsrt.sample_option(state, set(), rng)
    assert option.initiable(state)
    for _ in range(100):    # should terminate much earlier
        action = option.policy(state)
        ###
        # Update demos
        observation_list.append(obs)
        state_list.append(state)
        action_list.append(action)
        option_list.append((option.name, option.objects, option.params))
        ###
        obs = env.step(action)
        perceiver.update_perceiver_with_action(action)
        state = perceiver.step(obs)
        if option.terminal(state):
            break

    # Check that moving to the cube succeeded.
    assert GroundAtom(InHandView, [spot, cube]).holds(state)

    # Sample and run an option to pick cube from the table.
    grasp_cube_nsrt = PickObjectFromSide.ground([spot, cube, shelf])
    assert all(a.holds(state) for a in grasp_cube_nsrt.preconditions), f"{[(a, a.holds(state)) for a in grasp_cube_nsrt.preconditions]}"
    option = grasp_cube_nsrt.sample_option(state, set(), rng)
    assert option.initiable(state)
    for _ in range(100):    # should terminate much earlier
        action = option.policy(state)
        ###
        # Update demos
        observation_list.append(obs)
        state_list.append(state)
        action_list.append(action)
        option_list.append((option.name, option.objects, option.params))
        ###
        obs = env.step(action)
        perceiver.update_perceiver_with_action(action)
        state = perceiver.step(obs)
        if option.terminal(state):
            break

    # Check that picking succeeded.
    assert not GroundAtom(HandEmpty, [spot]).holds(state)
    assert GroundAtom(Holding, [spot, cube]).holds(state)

    # Sample and run an option to move to the other table.
    move_to_cabinet_nsrt = MoveToReachObject.ground([spot, cabinet])
    assert all(a.holds(state) for a in move_to_cabinet_nsrt.preconditions), f"{[(a, a.holds(state)) for a in move_to_cabinet_nsrt.preconditions]}"
    option = move_to_cabinet_nsrt.sample_option(state, set(), rng)
    assert option.initiable(state)
    for _ in range(100):    # should terminate much earlier
        action = option.policy(state)
        ###
        # Update demos
        observation_list.append(obs)
        state_list.append(state)
        action_list.append(action)
        option_list.append((option.name, option.objects, option.params))
        ###
        obs = env.step(action)
        perceiver.update_perceiver_with_action(action)
        state = perceiver.step(obs)
        if option.terminal(state):
            break

    # Check that moving to the table succeeded.
    assert GroundAtom(Reachable, [spot, cabinet])

    # Sample and run an option to place on the tables.
    place_cube_on_cabinet_nsrt = PlaceObjectOnTop.ground([spot, cube, cabinet])
    assert all(a.holds(state) for a in place_cube_on_cabinet_nsrt.preconditions), f"{[(a, a.holds(state)) for a in place_cube_on_cabinet_nsrt.preconditions]}"
    option = place_cube_on_cabinet_nsrt.sample_option(state, set(), rng)
    assert option.initiable(state)
    for _ in range(100):    # should terminate much earlier
        action = option.policy(state)
        ###
        # Update demos
        observation_list.append(obs)
        state_list.append(state)
        action_list.append(action)
        option_list.append((option.name, option.objects, option.params))
        ###
        obs = env.step(action)
        perceiver.update_perceiver_with_action(action)
        state = perceiver.step(obs)
        if option.terminal(state):
            break

    ###
    # Update demos
    observation_list.append(obs)
    state_list.append(state)
    ###

    # Check that placing on the table succeeded.
    assert GroundAtom(HandEmpty, [spot]).holds(state)
    assert not GroundAtom(Holding, [spot, cube]).holds(state)
    # assert GroundAtom(On, [cube, cabinet]).holds(state)

    ###
    # Save demo in pickle dict
    demo = {
        'observations': observation_list,
        'states': state_list,
        'actions': action_list,
        'options': option_list,
    }
    with open('spot_pickplace_demo.pkl', 'wb') as f:
        pickle.dump(demo, f)
    ###

if __name__ == '__main__':
    pick_and_place_script()