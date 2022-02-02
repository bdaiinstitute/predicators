"""Test cases for teacher."""

from predicators.src.envs import create_env
from predicators.src.ground_truth_nsrts import _get_predicates_by_names
from predicators.src.structs import GroundAtom, LowLevelTrajectory
from predicators.src import utils
from predicators.src.teacher import DemonstrationQuery, DemonstrationResponse,\
    Teacher, GroundAtomsHoldQuery, GroundAtomsHoldResponse


def test_GroundAtomsHold():
    """Tests for answering queries of type GroundAtomsHoldQuery."""
    utils.reset_config({"env": "cover", "approach": "unittest"})
    teacher = Teacher()
    env = create_env("cover")
    state = env.get_train_tasks()[0].init
    block_type = [t for t in env.types if t.name == "block"][0]
    target_type = [t for t in env.types if t.name == "target"][0]
    block = block_type("block0")
    target = target_type("target0")
    Covers, IsBlock = _get_predicates_by_names("cover", ["Covers", "IsBlock"])
    Covers = utils.strip_predicate(Covers)
    IsBlock = utils.strip_predicate(IsBlock)
    is_block_block = GroundAtom(IsBlock, [block])
    query = GroundAtomsHoldQuery({is_block_block})
    response = teacher.answer_query(state, query)
    assert isinstance(response, GroundAtomsHoldResponse)
    assert response.query is query
    assert len(response.holds) == 1
    assert response.holds[is_block_block]
    covers_block_target = GroundAtom(Covers, [block, target])
    query = GroundAtomsHoldQuery({covers_block_target})
    response = teacher.answer_query(state, query)
    assert isinstance(response, GroundAtomsHoldResponse)
    assert response.query is query
    assert len(response.holds) == 1
    assert not response.holds[covers_block_target]
    query = GroundAtomsHoldQuery({covers_block_target})
    response = teacher.answer_query(state, query)
    assert isinstance(response, GroundAtomsHoldResponse)
    assert response.query is query
    assert len(response.holds) == 1
    assert not response.holds[covers_block_target]
    query = GroundAtomsHoldQuery({is_block_block, covers_block_target})
    response = teacher.answer_query(state, query)
    assert isinstance(response, GroundAtomsHoldResponse)
    assert response.query is query
    assert len(response.holds) == 2
    assert response.holds[is_block_block]
    assert not response.holds[covers_block_target]


def test_DemonstrationQuery():
    """Tests for answering queries of type DemonstrationQuery."""
    utils.reset_config({"env": "cover", "approach": "unittest"})
    teacher = Teacher()
    env = create_env("cover")
    task = env.get_train_tasks()[0]
    state = task.init
    goal = task.goal
    # Test normal usage
    query = DemonstrationQuery(goal)
    response = teacher.answer_query(state, query)
    assert isinstance(response, DemonstrationResponse)
    assert response.query is query
    assert isinstance(response.teacher_traj, LowLevelTrajectory)
    print(len(response.teacher_traj.actions))
    assert len(response.teacher_traj.actions) == 2
    # (leverage GroundAtomsHoldQuery to test if the teacher_traj
    # successfully achieves the goal)
    goal_query = GroundAtomsHoldQuery(goal)
    goal_holds_response = teacher.answer_query(
        response.teacher_traj.states[-1], goal_query)
    assert False not in goal_holds_response.holds.items()
    # Test usage when goal is already achieved
    response = teacher.answer_query(response.teacher_traj.states[-1], query)
    assert isinstance(response, DemonstrationResponse)
    assert response.query is query
    assert isinstance(response.teacher_traj, LowLevelTrajectory)
    assert len(response.teacher_traj.actions) == 0
    # Test usage when achieving goal is impossible
    block_type = [t for t in env.types if t.name == "block"][0]
    block = block_type("block0")
    IsBlock = _get_predicates_by_names("cover", ["IsBlock"])[0]
    IsBlock = utils.strip_predicate(IsBlock)
    NotIsBlock = IsBlock.get_negation()
    not_is_block_block = GroundAtom(NotIsBlock, [block])
    query = DemonstrationQuery({not_is_block_block})
    response = teacher.answer_query(state, query)
    assert isinstance(response, DemonstrationResponse)
    assert response.query is query
    assert response.teacher_traj is None
