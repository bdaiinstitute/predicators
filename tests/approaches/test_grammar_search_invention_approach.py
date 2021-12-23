"""Test cases for the grammar search invention approach.
"""

import pytest
import numpy as np
from predicators.src.approaches.grammar_search_invention_approach import \
    _PredicateGrammar, _DataBasedPredicateGrammar, \
    _SingleFeatureInequalitiesPredicateGrammar, _count_positives_for_ops, \
    _create_grammar, _halving_constant_generator, _ForallClassifier, \
    _UnaryFreeForallClassifier, _create_score_function, \
    _PredicateSearchScoreFunction, _OperatorLearningBasedScoreFunction, \
    _HeuristicBasedScoreFunction, _HAddHeuristicBasedScoreFunction, \
    _HAddHeuristicMatchBasedScoreFunction, _PredictionErrorScoreFunction, \
    _HAddHeuristicLookaheadBasedScoreFunction, _TaskPlanningScoreFunction, \
    _ExactHeuristicLookaheadBasedScoreFunction, _BranchingFactorScoreFunction
from predicators.src.datasets import create_dataset
from predicators.src.envs import CoverEnv, BlocksEnv, PaintingEnv
from predicators.src.structs import Type, Predicate, STRIPSOperator, State, \
    Action, ParameterizedOption, Box, LowLevelTrajectory
from predicators.src.nsrt_learning import segment_trajectory
from predicators.src.settings import CFG
from predicators.src import utils


def test_predicate_grammar():
    """Tests for _PredicateGrammar class.
    """
    utils.update_config({"env": "cover"})
    env = CoverEnv()
    train_task = next(env.train_tasks_generator())[0]
    state = train_task.init
    other_state = state.copy()
    robby = [o for o in state if o.type.name == "robot"][0]
    state.set(robby, "hand", 0.5)
    other_state.set(robby, "hand", 0.8)
    dataset = [LowLevelTrajectory(
        [state, other_state], [np.zeros(1, dtype=np.float32)])]
    base_grammar = _PredicateGrammar()
    with pytest.raises(NotImplementedError):
        base_grammar.generate(max_num=1)
    data_based_grammar = _DataBasedPredicateGrammar(dataset)
    assert data_based_grammar.types == env.types
    with pytest.raises(NotImplementedError):
        data_based_grammar.generate(max_num=1)
    with pytest.raises(NotImplementedError):
        _create_grammar("not a real grammar name", dataset, set())
    env = CoverEnv()
    holding_dummy_grammar = _create_grammar("holding_dummy", dataset,
                                            env.predicates)
    assert len(holding_dummy_grammar.generate(max_num=1)) == 1
    assert len(holding_dummy_grammar.generate(max_num=3)) == 2
    single_ineq_grammar = _SingleFeatureInequalitiesPredicateGrammar(dataset)
    assert len(single_ineq_grammar.generate(max_num=1)) == 1
    feature_ranges = single_ineq_grammar._get_feature_ranges()  # pylint: disable=protected-access
    assert feature_ranges[robby.type]["hand"] == (0.5, 0.8)
    neg_sfi_grammar = _create_grammar("single_feat_ineqs", dataset,
                                      env.predicates)
    candidates = neg_sfi_grammar.generate(max_num=4)
    assert str(sorted(candidates)) == \
        ("[((0:block).pose<=2.33), ((0:block).width<=19.0), "
         "NOT-((0:block).pose<=2.33), NOT-((0:block).width<=19.0)]")
    forall_grammar = _create_grammar("forall_single_feat_ineqs", dataset,
                                     env.predicates)
    assert len(forall_grammar.generate(max_num=100)) == 100


def test_count_positives_for_ops():
    """Tests for _count_positives_for_ops().
    """
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    on = Predicate("On", [cup_type, plate_type], lambda s, o: True)
    not_on = Predicate("NotOn", [cup_type, plate_type], lambda s, o: True)
    cup_var = cup_type("?cup")
    plate_var = plate_type("?plate")
    parameters = [cup_var, plate_var]
    preconditions = {not_on([cup_var, plate_var])}
    add_effects = {on([cup_var, plate_var])}
    delete_effects = {not_on([cup_var, plate_var])}
    strips_operator = STRIPSOperator("Pick", parameters, preconditions,
                                     add_effects, delete_effects)
    cup = cup_type("cup")
    plate = plate_type("plate")
    parameterized_option = ParameterizedOption(
        "Dummy", [], Box(0, 1, (1,)),
        lambda s, m, o, p: Action(np.array([0.0])),
        lambda s, m, o, p: True, lambda s, m, o, p: True)
    option = parameterized_option.ground([], np.array([0.0]))
    state = State({cup: [0.5], plate: [1.0]})
    action = Action(np.zeros(1, dtype=np.float32))
    action.set_option(option)
    states = [state, state]
    actions = [action]
    strips_ops = [strips_operator]
    option_specs = [(parameterized_option, [])]
    pruned_atom_data = [
        # Test empty sequence.
        (LowLevelTrajectory([state], []), [{on([cup, plate])}]),
        # Test not positive.
        (LowLevelTrajectory(states, actions), [{on([cup, plate])}, set()]),
        # Test true positive.
        (LowLevelTrajectory(states, actions), [{not_on([cup, plate])},
                                               {on([cup, plate])}]),
        # Test false positive.
        (LowLevelTrajectory(states, actions), [{not_on([cup, plate])}, set()]),
    ]
    segments = [seg for traj in pruned_atom_data
                for seg in segment_trajectory(traj)]

    num_true, num_false, _, _ = _count_positives_for_ops(strips_ops,
         option_specs, segments)
    assert num_true == 1
    assert num_false == 1


def test_halving_constant_generator():
    """Tests for _halving_constant_generator().
    """
    expected_sequence = [0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875]
    generator = _halving_constant_generator(0., 1.)
    for i, x in zip(range(len(expected_sequence)), generator):
        assert abs(expected_sequence[i] - x) < 1e-6


def test_forall_classifier():
    """Tests for _ForallClassifier().
    """
    cup_type = Type("cup_type", ["feat1"])
    pred = Predicate("Pred", [cup_type],
        lambda s, o: s.get(o[0], "feat1") > 0.5)
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    state0 = State({cup1: [0.], cup2: [0.]})
    state1 = State({cup1: [0.], cup2: [1.]})
    state2 = State({cup1: [1.], cup2: [1.]})
    classifier = _ForallClassifier(pred)
    assert not classifier(state0, [])
    assert not classifier(state1, [])
    assert classifier(state2, [])
    assert str(classifier) == "Forall[0:cup_type].[Pred(0)]"


def test_unary_free_forall_classifier():
    """Tests for _UnaryFreeForallClassifier().
    """
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    on = Predicate("On", [cup_type, plate_type], lambda s, o: True)
    cup0 = cup_type("cup0")
    plate0 = plate_type("plate0")
    state0 = State({cup0: [0.], plate0: [0.]})
    classifier0 = _UnaryFreeForallClassifier(on, 0)
    assert classifier0(state0, [cup0])
    assert str(classifier0) == "Forall[1:plate_type].[On(0,1)]"
    classifier1 = _UnaryFreeForallClassifier(on, 1)
    assert classifier1(state0, [plate0])
    assert str(classifier1) == "Forall[0:cup_type].[On(0,1)]"


def test_create_score_function():
    """Tests for _create_score_function().
    """
    utils.update_config(
        {"grammar_search_score_function": "prediction_error"})
    score_func = _create_score_function(set(), [], [], {})
    assert isinstance(score_func, _PredictionErrorScoreFunction)
    utils.update_config(
        {"grammar_search_score_function": "hadd_match"})
    score_func = _create_score_function(set(), [], [], {})
    assert isinstance(score_func, _HAddHeuristicMatchBasedScoreFunction)
    utils.update_config(
        {"grammar_search_score_function": "branching_factor"})
    score_func = _create_score_function(set(), [], [], {})
    assert isinstance(score_func, _BranchingFactorScoreFunction)
    utils.update_config(
        {"grammar_search_score_function": "hadd_lookahead"})
    score_func = _create_score_function(set(), [], [], {})
    assert isinstance(score_func, _HAddHeuristicLookaheadBasedScoreFunction)
    utils.update_config(
        {"grammar_search_score_function": "exact_lookahead"})
    score_func = _create_score_function(set(), [], [], {})
    assert isinstance(score_func, _ExactHeuristicLookaheadBasedScoreFunction)
    utils.update_config(
        {"grammar_search_score_function": "task_planning"})
    score_func = _create_score_function(set(), [], [], {})
    assert isinstance(score_func, _TaskPlanningScoreFunction)
    utils.update_config(
        {"grammar_search_score_function": "not a real score function"})
    with pytest.raises(NotImplementedError):
        _create_score_function(set(), [], [], {})


def test_predicate_search_heuristic_base_classes():
    """Cover the abstract methods for _PredicateSearchScoreFunction & subclasses
    """
    pred_search_score_function = _PredicateSearchScoreFunction(
        set(), [], [], {})
    with pytest.raises(NotImplementedError):
        pred_search_score_function.evaluate(set())
    op_learning_score_function = _OperatorLearningBasedScoreFunction(
        set(), [], [], {})
    with pytest.raises(NotImplementedError):
        op_learning_score_function.evaluate(set())
    utils.update_config({"env": "cover"})
    env = CoverEnv()
    train_tasks = next(env.train_tasks_generator())
    state = train_tasks[0].init
    other_state = state.copy()
    robby = [o for o in state if o.type.name == "robot"][0]
    state.set(robby, "hand", 0.5)
    other_state.set(robby, "hand", 0.8)
    parameterized_option = ParameterizedOption(
        "Dummy", [], Box(0, 1, (1,)),
        lambda s, m, o, p: Action(np.array([0.0])),
        lambda s, m, o, p: True, lambda s, m, o, p: True)
    option = parameterized_option.ground([], np.array([0.0]))
    action = Action(np.zeros(1, dtype=np.float32))
    action.set_option(option)
    dataset = [LowLevelTrajectory(
        [state, other_state], [action], set())]
    atom_dataset = utils.create_ground_atom_dataset(dataset, set())
    heuristic_score_fn = _HeuristicBasedScoreFunction(
        set(), atom_dataset, train_tasks, {})
    with pytest.raises(NotImplementedError):
        heuristic_score_fn.evaluate(set())
    hadd_score_fn = _HAddHeuristicBasedScoreFunction(
        set(), atom_dataset, train_tasks, {})
    with pytest.raises(NotImplementedError):
        hadd_score_fn.evaluate(set())


def test_prediction_error_score_function():
    """Tests for _PredictionErrorScoreFunction().
    """
    # Tests for CoverEnv.
    utils.update_config({
        "env": "cover",
        "offline_data_method": "demo+replay",
        "seed": 0,
    })
    env = CoverEnv()
    ablated = {"HandEmpty", "Holding"}
    initial_predicates = set()
    name_to_pred = {}
    for p in env.predicates:
        if p.name in ablated:
            name_to_pred[p.name] = p
        else:
            initial_predicates.add(p)
    candidates = {p: 1.0 for p in name_to_pred.values()}
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)
    atom_dataset = utils.create_ground_atom_dataset(dataset, env.predicates)
    score_function = _PredictionErrorScoreFunction(
        initial_predicates, atom_dataset, train_tasks, candidates)
    all_included_s = score_function.evaluate(set(candidates))
    handempty_included_s = score_function.evaluate({name_to_pred["HandEmpty"]})
    holding_included_s = score_function.evaluate({name_to_pred["Holding"]})
    none_included_s = score_function.evaluate(set())
    assert all_included_s < holding_included_s < none_included_s
    assert all_included_s < handempty_included_s  # not better than none

    # Tests for BlocksEnv.
    utils.flush_cache()
    utils.update_config({
        "env": "blocks",
        "offline_data_method": "demo+replay",
        "seed": 0,
    })
    env = BlocksEnv()
    ablated = {"Holding", "Clear", "GripperOpen"}
    initial_predicates = set()
    name_to_pred = {}
    for p in env.predicates:
        if p.name in ablated:
            name_to_pred[p.name] = p
        else:
            initial_predicates.add(p)
    candidates = {p: 1.0 for p in name_to_pred.values()}
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)
    atom_dataset = utils.create_ground_atom_dataset(dataset, env.predicates)
    score_function = _PredictionErrorScoreFunction(
        initial_predicates, atom_dataset, train_tasks, candidates)
    all_included_s = score_function.evaluate(set(candidates))
    holding_included_s = score_function.evaluate({name_to_pred["Holding"]})
    clear_included_s = score_function.evaluate({name_to_pred["Clear"]})
    gripper_open_included_s = score_function.evaluate(
        {name_to_pred["GripperOpen"]})
    none_included_s = score_function.evaluate(set())
    assert all_included_s < holding_included_s < none_included_s
    assert all_included_s < clear_included_s < none_included_s
    assert all_included_s < gripper_open_included_s < none_included_s

    # Tests for PaintingEnv.
    utils.flush_cache()
    utils.update_config({
        "env": "painting",
        "offline_data_method": "demo+replay",
        "seed": 0,
        "painting_train_families": ["box_and_shelf"],
    })
    env = PaintingEnv()
    ablated = {"IsWet", "IsDry"}
    initial_predicates = set()
    name_to_pred = {}
    for p in env.predicates:
        if p.name in ablated:
            name_to_pred[p.name] = p
        else:
            initial_predicates.add(p)
    candidates = {p: 1.0 for p in name_to_pred.values()}
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)
    atom_dataset = utils.create_ground_atom_dataset(dataset, env.predicates)
    score_function = _PredictionErrorScoreFunction(
        initial_predicates, atom_dataset, train_tasks, candidates)
    all_included_s = score_function.evaluate(set(candidates))
    none_included_s = score_function.evaluate(set())
    assert all_included_s < none_included_s


def test_hadd_match_score_function():
    """Tests for _HAddHeuristicMatchBasedScoreFunction().
    """
    # We know that this score function is bad, and this test shows why.
    utils.update_config({
        "env": "cover",
        "offline_data_method": "demo+replay",
        "seed": 0,
    })
    env = CoverEnv()
    ablated = {"HandEmpty"}
    initial_predicates = set()
    name_to_pred = {}
    for p in env.predicates:
        if p.name in ablated:
            name_to_pred[p.name] = p
        else:
            initial_predicates.add(p)
    candidates = {p: 1.0 for p in name_to_pred.values()}
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)
    atom_dataset = utils.create_ground_atom_dataset(dataset, env.predicates)
    score_function = _HAddHeuristicMatchBasedScoreFunction(
        initial_predicates, atom_dataset, train_tasks, candidates)
    handempty_included_s = score_function.evaluate({name_to_pred["HandEmpty"]})
    none_included_s = score_function.evaluate(set())
    assert handempty_included_s > none_included_s # this is very bad!


def test_hadd_lookahead_score_function():
    """Tests for _HAddHeuristicLookaheadBasedScoreFunction().
    """
    # Tests for CoverEnv.
    utils.update_config({
        "env": "cover",
    })
    utils.update_config({
        "env": "cover",
        "offline_data_method": "demo+replay",
        "seed": 0,
    })
    env = CoverEnv()
    ablated = {"HandEmpty", "Holding"}
    initial_predicates = set()
    name_to_pred = {}
    for p in env.predicates:
        if p.name in ablated:
            name_to_pred[p.name] = p
        else:
            initial_predicates.add(p)
    candidates = {p: 1.0 for p in name_to_pred.values()}
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)
    atom_dataset = utils.create_ground_atom_dataset(dataset, env.predicates)
    score_function = _HAddHeuristicLookaheadBasedScoreFunction(
        initial_predicates, atom_dataset, train_tasks, candidates)
    all_included_s = score_function.evaluate(set(candidates))
    handempty_included_s = score_function.evaluate({name_to_pred["HandEmpty"]})
    holding_included_s = score_function.evaluate({name_to_pred["Holding"]})
    none_included_s = score_function.evaluate(set())
    assert all_included_s < holding_included_s < none_included_s
    assert all_included_s < handempty_included_s  # not better than none

    # Test that the score is inf when the operators make the data impossible.
    ablated = {"Covers"}
    initial_predicates = set()
    name_to_pred = {}
    for p in env.predicates:
        if p.name in ablated:
            name_to_pred[p.name] = p
        else:
            initial_predicates.add(p)
    candidates = {p: 1.0 for p in name_to_pred.values()}
    # Reuse dataset from above.
    score_function = _HAddHeuristicLookaheadBasedScoreFunction(
        initial_predicates, atom_dataset, train_tasks, candidates)
    assert score_function.evaluate(set()) == float("inf")

    # Tests for BlocksEnv.
    utils.flush_cache()
    utils.update_config({
        "env": "blocks",
        "offline_data_method": "demo+replay",
        "seed": 0,
    })
    env = BlocksEnv()
    ablated = {"Holding", "Clear", "GripperOpen"}
    initial_predicates = set()
    name_to_pred = {}
    for p in env.predicates:
        if p.name in ablated:
            name_to_pred[p.name] = p
        else:
            initial_predicates.add(p)
    candidates = {p: 1.0 for p in name_to_pred.values()}
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)
    atom_dataset = utils.create_ground_atom_dataset(dataset, env.predicates)
    score_function = _HAddHeuristicLookaheadBasedScoreFunction(
        initial_predicates, atom_dataset, train_tasks, candidates)
    all_included_s = score_function.evaluate(set(candidates))
    none_included_s = score_function.evaluate(set())
    gripperopen_excluded_s = score_function.evaluate({name_to_pred["Holding"],
                                                      name_to_pred["Clear"]})
    assert all_included_s < none_included_s  # good!
    # The fact that there is not a monotonic improvement shows a downside of
    # this score function. But we do see that learning works well in the end.
    assert gripperopen_excluded_s < all_included_s  # bad!
    # Note: here are all the scores.
    # (): 17640.461089410717
    # (Clear,): 21144.93016115656
    # (Holding,): 11240.237938078439
    # (GripperOpen,): 17641.505279500794
    # (Clear, Holding): 7581.118488743514
    # (Clear, GripperOpen): 21145.98910036367
    # (Holding, GripperOpen): 14643.702564367157
    # (Clear, Holding, GripperOpen): 11411.369394796291

    # Tests for PaintingEnv.
    utils.flush_cache()
    utils.update_config({
        "env": "painting",
        "offline_data_method": "demo+replay",
        "seed": 0,
        "painting_train_families": ["box_and_shelf"],
    })
    env = PaintingEnv()
    ablated = {"IsWet", "IsDry"}
    initial_predicates = set()
    name_to_pred = {}
    for p in env.predicates:
        if p.name in ablated:
            name_to_pred[p.name] = p
        else:
            initial_predicates.add(p)
    candidates = {p: 1.0 for p in name_to_pred.values()}
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)
    atom_dataset = utils.create_ground_atom_dataset(dataset, env.predicates)
    score_function = _HAddHeuristicLookaheadBasedScoreFunction(
        initial_predicates, atom_dataset, train_tasks, candidates)
    all_included_s = score_function.evaluate(set(candidates))
    none_included_s = score_function.evaluate(set())
    assert all_included_s < none_included_s  # hooray!


def test_exact_lookahead_score_function():
    """Tests for _ExactHeuristicLookaheadBasedScoreFunction().
    """
    # Just test this on BlocksEnv, since that's a known problem case
    # for hadd_lookahead.
    utils.flush_cache()
    utils.update_config({
        "env": "blocks",
    })
    utils.update_config({
        "env": "blocks",
        "offline_data_method": "demo+replay",
        "seed": 0,
        "num_train_tasks": 2,
    })
    env = BlocksEnv()
    ablated = {"Holding", "Clear", "GripperOpen"}
    initial_predicates = set()
    name_to_pred = {}
    for p in env.predicates:
        if p.name in ablated:
            name_to_pred[p.name] = p
        else:
            initial_predicates.add(p)
    candidates = {p: 1.0 for p in name_to_pred.values()}
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)
    atom_dataset = utils.create_ground_atom_dataset(dataset, env.predicates)
    score_function = _ExactHeuristicLookaheadBasedScoreFunction(
        initial_predicates, atom_dataset, train_tasks, candidates)
    all_included_s = score_function.evaluate(set(candidates))
    none_included_s = score_function.evaluate(set())
    gripperopen_excluded_s = score_function.evaluate({name_to_pred["Holding"],
                                                      name_to_pred["Clear"]})
    assert all_included_s < none_included_s  # good!
    assert all_included_s < gripperopen_excluded_s  # good!
    # Test that the score is inf when the operators make the data impossible.
    ablated = {"On"}
    initial_predicates = set()
    name_to_pred = {}
    for p in env.predicates:
        if p.name in ablated:
            name_to_pred[p.name] = p
        else:
            initial_predicates.add(p)
    candidates = {p: 1.0 for p in name_to_pred.values()}
    # Reuse dataset from above.
    score_function = _ExactHeuristicLookaheadBasedScoreFunction(
        initial_predicates, atom_dataset, train_tasks, candidates)
    assert score_function.evaluate(set()) == float("inf")
    old_hbmd = CFG.grammar_search_heuristic_based_max_demos
    utils.update_config({
        "grammar_search_heuristic_based_max_demos": 0})
    assert score_function.evaluate(set()) == 0.39  # only operator penalty
    utils.update_config({
        "grammar_search_heuristic_based_max_demos": old_hbmd})


def test_branching_factor_score_function():
    """Tests for _BranchingFactorScoreFunction().
    """
    # We know that this score function is bad, because it prefers predicates
    # that make segmentation collapse demo actions into one.
    utils.update_config({
        "env": "cover",
    })
    utils.update_config({
        "env": "cover",
        "offline_data_method": "demo+replay",
        "seed": 0,
    })
    env = CoverEnv()

    name_to_pred = {p.name : p for p in env.predicates}
    Covers = name_to_pred["Covers"]
    Holding = name_to_pred["Holding"]

    forall_not_covers0 = Predicate(
        "Forall[0:block].[NOT-Covers(0,1)]",
        [Covers.types[1]],
        _UnaryFreeForallClassifier(Covers.get_negation(), 1)
    )

    forall_not_covers1 = Predicate(
        "Forall[1:target].[NOT-Covers(0,1)]",
        [Covers.types[0]],
        _UnaryFreeForallClassifier(Covers.get_negation(), 0)
    )

    candidates = {
        forall_not_covers0: 1.0,
        forall_not_covers1: 1.0,
        Holding: 1.0,
    }
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)
    atom_dataset = utils.create_ground_atom_dataset(dataset,
        env.goal_predicates | set(candidates))
    score_function = _BranchingFactorScoreFunction(
        env.goal_predicates, atom_dataset, train_tasks, candidates)
    holding_s = score_function.evaluate({Holding})
    forall_not_covers_s = score_function.evaluate(
        {forall_not_covers0, forall_not_covers1})
    assert forall_not_covers_s > holding_s


def test_task_planning_score_function():
    """Tests for _TaskPlanningScoreFunction().
    """
    # We know that this score function is bad, because it's way too
    # optimistic: it thinks that any valid sequence of operators can
    # be refined into a plan. This unit test illustrates that pitfall.
    utils.update_config({
        "env": "cover",
    })
    utils.update_config({
        "env": "cover",
        "offline_data_method": "demo+replay",
        "seed": 0,
    })
    env = CoverEnv()

    name_to_pred = {p.name : p for p in env.predicates}
    Holding = name_to_pred["Holding"]
    HandEmpty = name_to_pred["HandEmpty"]

    candidates = {
        Holding: 1.0,
        HandEmpty: 1.0,
    }
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)
    atom_dataset = utils.create_ground_atom_dataset(dataset,
        env.goal_predicates | set(candidates))
    score_function = _TaskPlanningScoreFunction(
        env.goal_predicates, atom_dataset, train_tasks, candidates)
    all_included_s = score_function.evaluate({Holding, HandEmpty})
    none_included_s = score_function.evaluate(set())
    # This is terrible!
    assert none_included_s < all_included_s
    # Test cases where operators cannot plan to goal.
    utils.update_config({
        "min_data_for_nsrt": 10000,
    })
    assert score_function.evaluate(set()) == len(train_tasks) * 1e7
    # The +2 is for the cost of the two predicates.
    assert score_function.evaluate({Holding, HandEmpty}) == 2 + \
        len(train_tasks) * 1e7
    # Set this back to avoid screwing up other tests...
    utils.update_config({
        "min_data_for_nsrt": 3,
    })
