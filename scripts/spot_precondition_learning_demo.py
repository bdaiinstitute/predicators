import os
import sys
import pickle
import time

import numpy as np

from predicators import utils
from predicators.envs.spot_env import SpotCleanupShelfEnv
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.nsrt_learning.nsrt_learning_main import learn_new_nsrts_from_data
from predicators.perception.spot_perceiver import SpotPerceiver
from predicators.planning import generate_sas_file_for_fd, fd_plan_from_sas_file, PlanningFailure
from predicators.settings import CFG
from predicators.spot_utils.skills.spot_navigation import go_home
from predicators.structs import Action, GroundAtom, _GroundNSRT, LowLevelTrajectory, GroundAtomTrajectory, Task, ParameterizedOption, NSRT

def precondition_learning_script() -> None:
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
        "sampler_learner": "oracle",
        "segmenter": "spot",
        "sesame_task_planner": "fdsat"
    })
    rng = np.random.default_rng(123)
   
    env = SpotCleanupShelfEnv()
    perceiver = SpotPerceiver()
    options = get_gt_options(env.get_name())
    nsrts = get_gt_nsrts(env.get_name(), env.predicates, options)

    nsrt_name_to_nsrt = {n.name: n for n in nsrts}
    MoveToHandViewObject = nsrt_name_to_nsrt["MoveToHandViewObject"]
    MoveToReachObject = nsrt_name_to_nsrt["MoveToReachObject"]
    PickObjectFromSide = nsrt_name_to_nsrt["PickObjectFromSide"]
    PickObjectFromTop = nsrt_name_to_nsrt["PickObjectFromTop"]
    PlaceObjectOnTop = nsrt_name_to_nsrt["PlaceObjectOnTop"]
    DragToUnblockObject = nsrt_name_to_nsrt["DragToUnblockObject"]

    # The robot gripper should be empty, and the chair should be visible
    pred_name_to_pred = {p.name: p for p in env.predicates}
    HandEmpty = pred_name_to_pred["HandEmpty"]
    On = pred_name_to_pred["On"]
    NotOn = pred_name_to_pred["NotOn"]
    InHandView = pred_name_to_pred["InHandView"]
    Holding = pred_name_to_pred["Holding"]
    Blocking = pred_name_to_pred["Blocking"]
    Reachable = pred_name_to_pred["Reachable"]
    NotBlocked = pred_name_to_pred["NotBlocked"]
    TooHigh = pred_name_to_pred["TooHigh"]
    NotTooHigh = pred_name_to_pred["NotTooHigh"]

    ##############
    # First get the demo and do the precondition learning stuff
    with open('spot_pickplace_demo.pkl', 'rb') as f:
        demo = pickle.load(f)

    # Hack demo to make sure final state is correct
    for o in demo['states'][-1].data:
        if o.name == 'cube':
            demo['states'][-1].data[o][0] = 3.5
            demo['states'][-1].data[o][1] = -1.65
            demo['states'][-1].data[o][2] = -0.2
            demo['states'][-1].data[o][-3] = 0
            demo['states'][-1].data[o][-1] = 1

    for act, opt in zip(demo['actions'], demo['options']):
        option = nsrt_name_to_nsrt[opt[0]].option.ground(opt[1], opt[2])
        act.set_option(option)

    trajectories = [LowLevelTrajectory(demo['states'], demo['actions'], True, 0)]
    ground_atom_dataset = utils.create_ground_atom_dataset(trajectories, env.predicates)
    fixed_nsrts = set([n for n in nsrts if n.name != "PlaceObjectOnTop"])
    demo_cube = [obj for obj in demo['states'][0] if obj.name == "cube"][0]
    demo_cabinet = [obj for obj in demo['states'][0] if obj.name == "cabinet"][0]
    demo_spot = [obj for obj in demo['states'][0] if obj.name == "robot"][0]
    train_tasks = [Task(demo['states'][0], {GroundAtom(On, [demo_cube, demo_cabinet]), GroundAtom(HandEmpty, [demo_spot]), GroundAtom(NotTooHigh, [demo_spot, demo_cube])})]

    assert CFG.sampler_learner == "oracle"
    learned_nsrts, info = learn_new_nsrts_from_data(trajectories,
                                                    fixed_nsrts,
                                                    train_tasks,
                                                    env.predicates,
                                                    options,
                                                    env.action_space,
                                                    ground_atom_dataset,
                                                    sampler_learner=CFG.sampler_learner,
                                                    annotations=None)
    print('successfully ran learn_new_nsrts_from_data')

    predicate_probabilities = {}
    local_vars = set(info["local_vars_effects"] + info["local_vars_setup"])
    obj_to_var = info["obj_to_var"]
    print(obj_to_var)
    for atom in info["pre_image"]:
        # Only add predicates whose objects are all "local"
        if all(o in obj_to_var and obj_to_var[o] in local_vars for o in atom.objects):
            objects_changed_by_action = all(obj_to_var[o] in info['local_vars_effects'] for o in atom.objects)
            objects_changed_by_setup = all(obj_to_var[o] in info['local_vars_effects'] + info['local_vars_setup'] for o in atom.objects)
            objects_changed_by_setup_action_mixed = all(obj_to_var[o] in info['local_vars_setup'] for o in atom.objects)
            predicate_changed_unintended = atom in (info['added_effects'] - info['intended_added_effects'])
            predicate_changed_explained = atom in (info['explained_atoms'] & info['intended_added_effects'])
            predicate_changed_by_setup = atom in info['added_effects']

            # I'm missing here adding the delete effects, which is another good candidate for preconditions
            # Also, this is not the most efficient way to write this if-else, but it is somewhat more readable
            if predicate_changed_by_setup and not predicate_changed_unintended and not predicate_changed_explained:
                if objects_changed_by_action:
                    p = 0.75
                else:
                    p = 0.5
            elif predicate_changed_by_setup and not predicate_changed_unintended and predicate_changed_explained:
                if objects_changed_by_action:
                    p = 0.5
                else:
                    p = 0.25
            elif predicate_changed_by_setup and predicate_changed_unintended:
                if objects_changed_by_action:
                    p = 0.25
                else:
                    p = 0.125
            # The predicate was not changed by setup
            elif objects_changed_by_action and objects_changed_by_setup:
                p = 0.5
            elif objects_changed_by_action:
                p = 0.25
            elif objects_changed_by_setup or objects_changed_by_setup_action_mixed:
                p = 0.125
            else:
                # print('objects_changed_by_action:', objects_changed_by_action)
                # print('objects_changed_by_setup:', objects_changed_by_setup)
                # print('predicate_changed_unintended:', predicate_changed_unintended)
                # print('predicate_changed_explained:', predicate_changed_explained)
                # print('predicate_changed_by_setup:', predicate_changed_by_setup)
                raise ValueError("Seems like I didn't consider all cases :-(")
            predicate_probabilities[atom] = p
    predicate_probabilities = {atom.lift(obj_to_var): prob for atom, prob in sorted(predicate_probabilities.items(), key=lambda x: x[1], reverse=True)}
    print('(',len(predicate_probabilities), '):', predicate_probabilities)
    learned_nsrt = next(iter(learned_nsrts))  # It should be a single learned NSRT, so use this hack to get it
    print(learned_nsrt.sampler)
    ##############


    ##############
    # Then check that the atoms that we care about in the initial state hold
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
    assert GroundAtom(HandEmpty, [spot]).holds(state)
    assert GroundAtom(On, [cube, shelf]).holds(state)
    assert GroundAtom(Blocking, [chair, cabinet]).holds(state)
    assert GroundAtom(TooHigh, [spot, cube]).holds(state)
    ##############

    ##############
    # Then get the plan skeletons per precondition set and determine which to try
    curr_preconditions = set()
    goal = {GroundAtom(NotOn, [cube, shelf]), GroundAtom(NotOn, [cube, floor]), GroundAtom(HandEmpty, [spot])}
    init_atoms = utils.abstract(state, env.predicates)
    objects = list(state)
    task = Task(state, goal)

    len_success_list = []
    len_fail_list = []
    p_success_list = []
    skeleton_list = []
    p_success = 0
    learned_nsrt = next(iter(learned_nsrts))  # It should be a single learned NSRT, so use this hack to get it
    next_predicate = 0
    atoms = init_atoms
    while True:
        curr_parameters = {o for atom in curr_preconditions | learned_nsrt.add_effects | learned_nsrt.delete_effects for o in atom.variables}
        curr_parameters |= set(learned_nsrt.option_vars)
        curr_nsrt = NSRT(learned_nsrt.name, curr_parameters, curr_preconditions, learned_nsrt.add_effects,
                         learned_nsrt.delete_effects, learned_nsrt.ignore_effects, learned_nsrt.option,
                         learned_nsrt.option_vars, learned_nsrt.sampler)

        try:
            skeleton, atoms_sequence = _generate_skeleton_via_fast_downward(
                task,
                atoms,
                objects,
                fixed_nsrts | {curr_nsrt},
                env.predicates,
                env.types,
                CFG.timeout,
                CFG.seed,
                max_horizon=CFG.horizon,
                optimal=CFG.sesame_task_planner == "fdopt")
        except PlanningFailure:
            ##################
            # First navigate to floor
            navigate_to_floor = MoveToReachObject.ground((spot, floor))
            assert navigate_to_floor.preconditions.issubset(atoms)
            atoms = utils.apply_operator(navigate_to_floor, atoms)

            # Then place on floor
            place_on_floor = curr_nsrt.ground((shelf, floor, spot, cube))
            assert place_on_floor.preconditions.issubset(atoms)
            atoms = utils.apply_operator(place_on_floor, atoms)

            # Then navigate to chair
            navigate_to_chair = MoveToHandViewObject.ground((spot, chair))
            assert navigate_to_chair.preconditions.issubset(atoms)
            atoms = utils.apply_operator(navigate_to_chair, atoms)

            # Then grasp chair
            grasp_chair = PickObjectFromTop.ground((spot, chair, floor))
            assert grasp_chair.preconditions.issubset(atoms)
            atoms = utils.apply_operator(grasp_chair, atoms)

            # Then drag chair
            drag_chair = DragToUnblockObject.ground((spot, cabinet, chair))
            assert drag_chair.preconditions.issubset(atoms)
            atoms = utils.apply_operator(drag_chair, atoms)

            # Then navigate to cube
            navigate_to_cube = MoveToHandViewObject.ground((spot, cube))
            assert navigate_to_cube.preconditions.issubset(atoms)
            atoms = utils.apply_operator(navigate_to_cube, atoms)

            # Then pick cube
            pick_cube = PickObjectFromTop.ground((spot, cube, floor))
            assert pick_cube.preconditions.issubset(atoms), f"{pick_cube.preconditions - atoms}"
            atoms = utils.apply_operator(pick_cube, atoms)

            # Then navigate to cabinet
            navigate_to_cabinet = MoveToReachObject.ground((spot, cabinet))
            assert navigate_to_cabinet.preconditions.issubset(atoms)
            atoms = utils.apply_operator(navigate_to_cabinet, atoms)

            # Then place on cabinet
            place_on_cabinet = curr_nsrt.ground((shelf, cabinet, spot, cube))
            assert place_on_cabinet.preconditions.issubset(atoms)
            atoms = utils.apply_operator(place_on_cabinet, atoms)

            assert goal.issubset(atoms)
            ##################

            raise
            print('removing:', new_precondition)
            curr_preconditions.remove(new_precondition)
            predicate_probabilities[new_precondition] = 0
            if next_predicate == len(predicate_probabilities):
                break
            new_precondition = list(predicate_probabilities.keys())[next_predicate]
            curr_preconditions.add(new_precondition)
            print()
            print('adding:', new_precondition)
            next_predicate += 1
            continue
        skeleton_list.append(skeleton)
        len_success_list.append(len(skeleton))
        for i in range(len(skeleton)):
            if skeleton[i].name == learned_nsrt.name:
                break
            atoms = utils.apply_operator(skeleton[i], atoms)
        else:   # never had to try new NSRT, so skip learning or something
            print("Never had to try the new NSRT. Quitting.")
            return
        len_fail_list.append(i + 1)
        # TODO: the line below assumes that the probability of success only depends on the atoms that were explicitly in the
        # preconditions, but doesn't check if those atoms are true anyway -- this is because we don't have a good way to
        # find a matching from objects to variables (see my own notes from [2023-08-29])
        p_success = np.prod([1 - prob for atom, prob in predicate_probabilities.items() if atom not in curr_preconditions])
        print('p succes:', p_success)
        p_success_list.append(p_success)

        print('skeleton:')
        for nsrt in skeleton:
            print('\t', nsrt.name, nsrt.objects)
        print()
        if next_predicate == len(predicate_probabilities):
            break
        new_precondition = list(predicate_probabilities.keys())[next_predicate]
        curr_preconditions.add(new_precondition)
        print()
        print('adding:', new_precondition)
        next_predicate += 1

    costs = [len_success_list[-1]]  # assume including all predicates leads to 100% p_success
    decisions = []


    curr_predicate_probabilities = {k: v for k, v in predicate_probabilities.items() if v > 0}

    # for i, (pred, len_success, len_fail, p_success) in reversed(list(enumerate(zip(curr_predicate_probabilities, len_success_list, len_fail_list, p_success_list)))):
    for i in range(len(curr_predicate_probabilities) - 1, -1, -1):
        pred = list(curr_predicate_probabilities.keys())[i]
        len_success = len_success_list[i + 1]
        len_fail = len_fail_list[i + 1]
        p_success = p_success_list[i + 1]
        skeleton = skeleton_list[i + 1]

        # Costs to try/not try without including the i-th predicate in the precondition set
        cost_try_succeed = len_success
        cost_try_fail = len_fail + costs[-1]
        cost_try = p_success * cost_try_succeed + (1 - p_success) * cost_try_fail
        cost_not_try = len_fail - 1 + costs[-1]

        if True or first_time_learning: # self._online_learning_cycle == 0
            # In the first loop, try everything
            if skeleton[0].name != learned_nsrt.name:
                costs.append(cost_try)
                decisions.append(True)
            else:
                costs.append(cost_not_try)
                decisions.append(False)
        elif cost_try < cost_not_try:
            costs.append(cost_try)
            decisions.append(True)
        else:
            costs.append(cost_not_try)
            decisions.append(False)
        print(pred, predicate_probabilities[pred], len_success, len_fail, p_success, cost_try, cost_not_try, decisions[-1])
    decisions = decisions[::-1]

    ### TODO Run low-level search on "longest" plan? Or on "current" plan in for loop?
    # For now, I'll just assume that we can always low-level search and there's no dependency between before and after the new action
    curr_preconditions = set()
    failed_once = False
    for pred, try_without in zip(curr_predicate_probabilities, decisions):
        # Hack to "seed" the learner with these two preconditionss
        if pred.predicate.name == "Reachable" or pred.predicate.name == "Holding":
            try_without = False
        print(pred, try_without)

        if try_without:
            curr_parameters = {o for atom in curr_preconditions | learned_nsrt.add_effects | learned_nsrt.delete_effects for o in atom.variables}
            curr_parameters |= set(learned_nsrt.option_vars)
            curr_nsrt = NSRT(learned_nsrt.name, curr_parameters, curr_preconditions, learned_nsrt.add_effects,
                             learned_nsrt.delete_effects, learned_nsrt.ignore_effects, learned_nsrt.option,
                             learned_nsrt.option_vars, learned_nsrt.sampler)
            skeleton, atoms_sequence = _generate_skeleton_via_fast_downward(
                task,
                init_atoms,
                objects,
                fixed_nsrts | {curr_nsrt},
                env.predicates,
                env.types,
                CFG.timeout,
                CFG.seed,
                max_horizon=CFG.horizon,
                optimal=CFG.sesame_task_planner == "fdopt")
            partial_skeleton = []
            partial_atoms_sequence = [atoms_sequence[0]]
            for i in range(len(skeleton)):
                partial_skeleton.append(skeleton[i])
                partial_atoms_sequence.append(atoms_sequence[i+1])
                if skeleton[i].name == learned_nsrt.name:
                    break

            state, suc = run_skeleton_on_spot(env, state, perceiver, partial_skeleton, partial_atoms_sequence, rng)
            init_atoms = utils.abstract(state, env.predicates)   # for next round
            if suc:
                print("Succeeded! Using:", curr_preconditions)
                break

        # Grow the precondition set
        curr_preconditions.add(pred)
    else:   # didn't succeed yet, repeat
        curr_parameters = {o for atom in curr_preconditions | learned_nsrt.add_effects | learned_nsrt.delete_effects for o in atom.variables}
        curr_parameters |= set(learned_nsrt.option_vars)
        curr_nsrt = NSRT(learned_nsrt.name, curr_parameters, curr_preconditions, learned_nsrt.add_effects,
                         learned_nsrt.delete_effects, learned_nsrt.ignore_effects, learned_nsrt.option,
                         learned_nsrt.option_vars, learned_nsrt.sampler)
        skeleton, atoms_sequence = _generate_skeleton_via_fast_downward(
            task,
            init_atoms,
            objects,
            fixed_nsrts | {curr_nsrt},
            env.predicates,
            env.types,
            CFG.timeout,
            CFG.seed,
            max_horizon=CFG.horizon,
            optimal=CFG.sesame_task_planner == "fdopt")
        partial_skeleton = []
        partial_atoms_sequence = [atoms_sequence[0]]
        for i in range(len(skeleton)):
            partial_skeleton.append(skeleton[i])
            partial_atoms_sequence.append(atoms_sequence[i+1])
            if skeleton[i].name == learned_nsrt.name:
                break

        state, suc = run_skeleton_on_spot(env, state, perceiver, partial_skeleton, partial_atoms_sequence, rng)
        if suc:
            print("Succeeded! Using:", curr_preconditions)

    ## I don't think we need anything beyond this.
    ##############


def _generate_skeleton_via_fast_downward(
    task, init_atoms, objects, nsrts, predicates, types, 
    timeout, seed, max_horizon, optimal):  # pragma: no cover 
    timeout_cmd = "gtimeout" if sys.platform == "darwin" else "timeout"
    if optimal:
        print("running optimal task planning")
        alias_flag = "--alias seq-opt-lmcut"
    else:  # satisficing
        alias_flag = "--alias lama-first"
        # alias_flag = "--alias seq-sat-lama-2011"
    # Run Fast Downward followed by cleanup. Capture the output.
    assert "FD_EXEC_PATH" in os.environ, \
        "Please follow the instructions in the docstring of this method!"
    fd_exec_path = os.environ["FD_EXEC_PATH"]
    exec_str = os.path.join(fd_exec_path, "fast-downward.py")
    start_time = time.perf_counter()
    sas_file = generate_sas_file_for_fd(task, nsrts, predicates, types,
                                        timeout, timeout_cmd, alias_flag,
                                        exec_str, objects, init_atoms)

    skeleton, atoms_sequence, metrics = fd_plan_from_sas_file(
        sas_file, timeout_cmd, timeout, exec_str, alias_flag, start_time,
        objects, init_atoms, nsrts, max_horizon)


    necessary_atoms_seq = utils.compute_necessary_atoms_seq(
            skeleton, atoms_sequence, task.goal)

    skeleton, necessary_atoms_seq = utils.trim_skeleton_to_necessary_atoms(
            skeleton, necessary_atoms_seq)

    return skeleton, necessary_atoms_seq

def run_skeleton_on_spot(env, state, perceiver, partial_skeleton, partial_atoms_sequence, rng):
    print("Skeleton to try:")   
    for nsrt in partial_skeleton:
        print(nsrt, nsrt.objects)
    for nsrt, expected_atoms in zip(partial_skeleton, partial_atoms_sequence[1:]):
        print("Attempting to execute", nsrt)
        print()
        assert all(a.holds(state) for a in nsrt.preconditions)
        option = nsrt.sample_option(state, set(), rng)
        assert option.initiable(state)
        for _ in range(100):    # should terminate much earlier
            action = option.policy(state)
            obs = env.step(action)
            perceiver.update_perceiver_with_action(action)
            state = perceiver.step(obs)
            if option.terminal(state):
                break
        if any(not a.holds(state) for a in expected_atoms):
            print([(a, a.holds(state)) for a in expected_atoms])
            return state, False

    return state, True



if __name__ == "__main__":
    precondition_learning_script()