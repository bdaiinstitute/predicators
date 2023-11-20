import numpy as np

from predicators.envs.spot_env import SpotCleanupShelfEnv


def drag_chair_script() -> None:
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
    nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                         get_gt_options(env.get_name))

    # This should have the robot spin around and locate all objects.
    task = env.get_test_tasks()[0]
    obs = env.reset("test", 0)
    perceiver.reset(task)
    cube, floor, table1, table2 = sorted(obs.objects_in_view)
    assert cube.name == "cube"
    assert "table" in table1.name
    assert "table" in table2.name
    state = perceiver.step(obs)
    spot = next(o for o in state if o.type.name == "robot")
    nsrt_name_to_nsrt = {n.name: n for n in nsrts}
    MoveToToolOnSurface = nsrt_name_to_nsrt["MoveToHandViewObject"]#["MoveToToolOnSurface"]
    MoveToSurface = nsrt_name_to_nsrt["MoveToReachObject"]#["MoveToSurface"]
    ground_nsrts: List[_GroundNSRT] = []
    for nsrt in sorted(nsrts):
        ground_nsrts.extend(utils.all_ground_nsrts(nsrt, set(state)))

    # The robot gripper should be empty, and the chair should be visible
    pred_name_to_pred = {p.name: p for p in env.predicates}
    HandEmpty = pred_name_to_pred["HandEmpty"]
    