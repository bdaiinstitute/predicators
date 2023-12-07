import numpy as np

from predicators import utils
from predicators.envs.spot_env import SpotCleanupShelfEnv
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.perception.spot_perceiver import SpotPerceiver
from predicators.settings import CFG
from predicators.spot_utils.skills.spot_navigation import go_home
from predicators.structs import Action, GroundAtom, _GroundNSRT


def drag_chair_script() -> None:
    args = utils.parse_args(env_required=False,
                            seed_required=False,
                            approach_required=False)
    utils.reset_config({
        "env": "spot_cleanup_shelf_env",
        "approach": "spot_wrapper[oracle]",
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "seed": 123,
        "spot_robot_ip": args["spot_robot_ip"],
        "test_task_json_dir": args.get("test_task_json_dir", None),
    })
    rng = np.random.default_rng(123)
    env = SpotCleanupShelfEnv()
    perceiver = SpotPerceiver()
    nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                         get_gt_options(env.get_name()))

    # This should have the robot spin around and locate all objects.
    task = env.get_test_tasks()[0]
    obs = env.reset("test", 0)
    perceiver.reset(task)
    print(obs.objects_in_view)
    cabinet, chair, cube, floor, shelf, yoga_ball = sorted(obs.objects_in_view)
    assert cabinet.name == "cabinet"
    assert chair.name == "chair"
    assert cube.name == "cube"
    assert floor.name == "floor"
    assert shelf.name == "shelf"
    assert yoga_ball.name == "yoga_ball"

    state = perceiver.step(obs)
    spot = next(o for o in state if o.type.name == "robot")
    nsrt_name_to_nsrt = {n.name: n for n in nsrts}
    MoveToHandViewObject = nsrt_name_to_nsrt["MoveToHandViewObject"]
    MoveToReachObject = nsrt_name_to_nsrt["MoveToReachObject"]
    PickObjectFromSide = nsrt_name_to_nsrt["PickObjectFromSide"]
    PickObjectFromTop = nsrt_name_to_nsrt["PickObjectFromTop"]
    PlaceObjectOnTop = nsrt_name_to_nsrt["PlaceObjectOnTop"]

    # The robot gripper should be empty, and the chair should be visible
    pred_name_to_pred = {p.name: p for p in env.predicates}
    HandEmpty = pred_name_to_pred["HandEmpty"]
    On = pred_name_to_pred["On"]
    InHandView = pred_name_to_pred["InHandView"]
    Holding = pred_name_to_pred["Holding"]
    Blocking = pred_name_to_pred["Blocking"]
    Reachable = pred_name_to_pred["Reachable"]
    NotBlocked = pred_name_to_pred["NotBlocked"]
    assert GroundAtom(HandEmpty, [spot]).holds(state)
    assert GroundAtom(On, [cube, shelf]).holds(state)
    assert GroundAtom(Blocking, [chair, cabinet]).holds(state)

    #### First cube move
    # Sample and run an option to move to the cube
    move_to_cube_nsrt = MoveToHandViewObject.ground((spot, cube))
    assert all(a.holds(state) for a in move_to_cube_nsrt.preconditions), f"{[(a, a.holds(state)) for a in move_to_cube_nsrt.preconditions]}"
    option = move_to_cube_nsrt.sample_option(state, set(), rng)
    assert option.initiable(state)
    for _ in range(100):    # should terminate much earlier
        action = option.policy(state)
        obs = env.step(action)
        perceiver.update_perceiver_with_action(action)
        state = perceiver.step(obs)
        if option.terminal(state):
            break

    # Check that moving to the cube succeeded.
    assert GroundAtom(InHandView, [spot, cube]).holds(state)

    # Sample and run an option to pick cube from the shelf.
    grasp_cube_nsrt = PickObjectFromSide.ground([spot, cube, shelf])
    assert all(a.holds(state) for a in grasp_cube_nsrt.preconditions), f"{[(a, a.holds(state)) for a in grasp_cube_nsrt.preconditions]}"
    option = grasp_cube_nsrt.sample_option(state, set(), rng)
    assert option.initiable(state)
    for _ in range(100):    # should terminate much earlier
        action = option.policy(state)
        obs = env.step(action)
        perceiver.update_perceiver_with_action(action)
        state = perceiver.step(obs)
        if option.terminal(state):
            break

    # Check that picking succeeded.
    assert not GroundAtom(HandEmpty, [spot]).holds(state)
    assert GroundAtom(Holding, [spot, cube]).holds(state)

    # Sample and run an option to move to the unstable surface.
    move_to_ball_nsrt = MoveToReachObject.ground([spot, yoga_ball])
    assert all(a.holds(state) for a in move_to_ball_nsrt.preconditions), f"{[(a, a.holds(state)) for a in move_to_ball_nsrt.preconditions]}"
    option = move_to_ball_nsrt.sample_option(state, set(), rng)
    assert option.initiable(state)
    for _ in range(100):    # should terminate much earlier
        action = option.policy(state)
        obs = env.step(action)
        perceiver.update_perceiver_with_action(action)
        state = perceiver.step(obs)
        if option.terminal(state):
            break

    # Check that moving to the ball succeeded.
    assert GroundAtom(Reachable, [spot, yoga_ball])

    # Sample and run an option to place on the ball.
    place_cube_on_ball_nsrt = PlaceObjectOnTop.ground([spot, cube, yoga_ball])
    assert all(a.holds(state) for a in place_cube_on_ball_nsrt.preconditions), f"{[(a, a.holds(state)) for a in place_cube_on_ball_nsrt.preconditions]}"
    option = place_cube_on_ball_nsrt.sample_option(state, set(), rng)
    assert option.initiable(state)
    for _ in range(100):    # should terminate much earlier
        action = option.policy(state)
        obs = env.step(action)
        perceiver.update_perceiver_with_action(action)
        state = perceiver.step(obs)
        if option.terminal(state):
            break

    # Check that placing on the ball failed.
    assert GroundAtom(HandEmpty, [spot]).holds(state)
    assert not GroundAtom(Holding, [spot, cube]).holds(state)
    assert not GroundAtom(On, [cube, yoga_ball]).holds(state)
    # assert GroundAtom(On, [cube, floor]).holds(state)
    #### End first cube move

    #### Move chair
    # Sample and run an option to move to the chair.
    move_to_chair_nsrt = MoveToHandViewObject.ground((spot, chair))
    option = move_to_chair_nsrt.sample_option(state, set(), rng)
    assert option.initiable(state)
    for _ in range(100):    # should terminate much earlier
        action = option.policy(state)
        obs = env.step(action)
        perceiver.update_perceiver_with_action(action)
        state = perceiver.step(obs)
        if option.terminal(state):
            break

    # Check that moving to the chair succeeded.
    assert GroundAtom(InHandView, [spot, chair]).holds(state)

    # Sample and run an option to pick chair from the floor.
    grasp_chair_nsrt = PickObjectFromTop.ground([spot, chair, floor])
    assert all(a.holds(state) for a in grasp_chair_nsrt.preconditions), f"{[(a, a.holds(state)) for a in grasp_chair_nsrt.preconditions]}"
    option = grasp_chair_nsrt.sample_option(state, set(), rng)
    print("Name of grasp nsrt and option:", grasp_chair_nsrt.name, option.name)
    assert option.initiable(state)
    for _ in range(100):    # should terminate much earlier
        action = option.policy(state)
        obs = env.step(action)
        perceiver.update_perceiver_with_action(action)
        state = perceiver.step(obs)
        if option.terminal(state):
            break

    # Check that picking succeeded.
    assert not GroundAtom(HandEmpty, [spot]).holds(state)
    assert GroundAtom(Holding, [spot, chair]).holds(state)

    # Sample and run an option to drag the chair.
    DragToUnblockObject = nsrt_name_to_nsrt["DragToUnblockObject"]
    unblock_cabinet_nsrt = DragToUnblockObject.ground([spot, cabinet, chair])
    assert all(a.holds(state) for a in unblock_cabinet_nsrt.preconditions), f"{[(a, a.holds(state)) for a in unblock_cabinet_nsrt.preconditions]}"
    option = unblock_cabinet_nsrt.sample_option(state, set(), rng)
    assert option.initiable(state)
    for _ in range(100):    # should terminate much earlier
        action = option.policy(state)
        obs = env.step(action)
        perceiver.update_perceiver_with_action(action)
        state = perceiver.step(obs)
        if option.terminal(state):
            break

    # Check that unblocking succeeded.
    assert GroundAtom(NotBlocked, [cabinet])
    #### End move chair

    #### Second cube move
    # Sample and run an option to move to the cube.
    move_to_cube_nsrt = MoveToHandViewObject.ground((spot, cube))
    option = move_to_cube_nsrt.sample_option(state, set(), rng)
    assert option.initiable(state)
    for _ in range(100):    # should terminate much earlier
        action = option.policy(state)
        obs = env.step(action)
        perceiver.update_perceiver_with_action(action)
        state = perceiver.step(obs)
        if option.terminal(state):
            break

    # Check that moving to the cube succeeded.
    assert GroundAtom(InHandView, [spot, cube]).holds(state)

    # Sample and run an option to pick from the floor
    grasp_cube_nsrt = PickObjectFromTop.ground([spot, cube, floor])
    assert all(a.holds(state) for a in grasp_cube_nsrt.preconditions), f"{[(a, a.holds(state)) for a in grasp_cube_nsrt.preconditions]}"
    option = grasp_cube_nsrt.sample_option(state, set(), rng)
    assert option.initiable(state)
    for _ in range(100):    # should terminate much earlier
        action = option.policy(state)
        obs = env.step(action)
        perceiver.update_perceiver_with_action(action)
        state = perceiver.step(obs)
        if option.terminal(state):
            break

    # Check that picking succeeded.
    assert not GroundAtom(HandEmpty, [spot]).holds(state)
    assert GroundAtom(Holding, [spot, cube]).holds(state)

    # Sample and run an option to move to the cabinet.
    move_to_cabinet_nsrt = MoveToReachObject.ground([spot, cabinet])
    assert all(a.holds(state) for a in move_to_ball_nsrt.preconditions), f"{[(a, a.holds(state)) for a in move_to_ball_nsrt.preconditions]}"
    option = move_to_cabinet_nsrt.sample_option(state, set(), rng)
    assert option.initiable(state)
    for _ in range(100):    # should terminate much earlier
        action = option.policy(state)
        obs = env.step(action)
        perceiver.update_perceiver_with_action(action)
        state = perceiver.step(obs)
        if option.terminal(state):
            break

    # Check that moving to the cabinet succeeded.
    assert GroundAtom(Reachable, [spot, cabinet])

    # Sample and run an option to place on the cabinet.
    place_cube_on_cabinet_nsrt = PlaceObjectOnTop.ground([spot, cube, cabinet])
    assert all(a.holds(state) for a in place_cube_on_cabinet_nsrt.preconditions), f"{[(a, a.holds(state)) for a in place_cube_on_cabinet_nsrt.preconditions]}"
    option = place_cube_on_cabinet_nsrt.sample_option(state, set(), rng)
    assert option.initiable(state)
    for _ in range(100):    # should terminate much earlier
        action = option.policy(state)
        obs = env.step(action)
        perceiver.update_perceiver_with_action(action)
        state = perceiver.step(obs)
        if option.terminal(state):
            break

    # Check that placing on the cabinet succeded.
    assert GroundAtom(HandEmpty, [spot]).holds(state)
    assert not GroundAtom(Holding, [spot, cube]).holds(state)
    assert GroundAtom(On, [cube, cabinet]).holds(state)
    #### End second cube move

if __name__ == '__main__':
    drag_chair_script()