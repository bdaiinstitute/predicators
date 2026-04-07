#!/usr/bin/env python3
"""Utility to regenerate mock Spot transition visualizations.

This recreates the HTML / PNG / SVG graph artifacts for a mock environment
without rerunning the full manual image-mapping workflow.
"""

import argparse
import importlib
import sys
from pathlib import Path

# Allow direct execution via `python scripts/regenerate_mock_env_graphs.py ...`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predicators import utils
from predicators.spot_utils.mock_env.mock_env_creator_base import MockEnvCreatorBase


def _load_env_class(env_name: str):
    """Return the mock Spot environment class by name."""
    # Try importing from mock_spot_env first
    module = importlib.import_module("predicators.envs.mock_spot_env")
    env_cls = getattr(module, env_name, None)

    # If not found, try pomdp_pddl_env
    if env_cls is None:
        try:
            module_pddl = importlib.import_module("predicators.envs.pomdp_pddl_env")
            env_cls = getattr(module_pddl, env_name, None)
        except ImportError:
            pass

    if env_cls is None:
        available = sorted(
            name
            for name in dir(module)
            if name.startswith("MockSpot") and name.endswith("Env")
        )
        raise ValueError(
            f"Unknown environment '{env_name}'. "
            f"Available mock env classes: {', '.join(available)}"
        )
    return env_cls


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate transition graph visualizations for a " "mock Spot environment."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mock_env_data"),
        help=(
            "Root directory that stores mock environment "
            "artifacts (defaults to ./mock_env_data)."
        ),
    )
    parser.add_argument(
        "--env-name",
        required=True,
        help="Mock environment class name (e.g., MockSpotCupEmptinessEnv).",
    )
    parser.add_argument(
        "--task-title",
        default=None,
        help=(
            "Optional override for the graph title. "
            "Defaults to 'Transition Graph, <Env Name>'."
        ),
    )
    args = parser.parse_args()

    try:
        env_class = _load_env_class(args.env_name)
    except ValueError as exc:  # pragma: no cover - CLI guard
        print(exc)
        sys.exit(1)

    # Reset CFG with minimal settings so env construction behaves consistently.
    utils.reset_config(
        {
            "seed": 0,
            "approach": "oracle",
            "env": env_class.__name__.lower(),
            "num_train_tasks": 0,
            "num_test_tasks": 1,
            "perceiver": "mock_spot_perceiver",
            "mock_env_vlm_eval_predicate": True,
            "bilevel_plan_without_sim": True,
            "horizon": 20,
        }
    )

    env = env_class(use_gui=False)
    env_dir = args.output_dir / env_class.__name__
    env_dir.mkdir(parents=True, exist_ok=True)

    creator = MockEnvCreatorBase(env_dir, env=env)

    # Rebuild the graph data and regenerate the Cytoscape/Graphviz outputs.
    creator.explore_states(env.initial_atoms, env.objects)
    task_title = (
        args.task_title or f"Transition Graph, {env.name.replace('_', ' ').title()}"
    )
    import json

    belief_log = []
    creator.plan_and_visualize(
        env.initial_atoms,
        env.goal_atoms_or,
        env.objects,
        task_name=task_title,
        belief_viz_log=belief_log,
    )

    # Save belief log
    if belief_log:
        belief_log_path = env_dir / "belief_log.json"
        with open(belief_log_path, "w") as f:
            json.dump(belief_log, f, indent=2)
        print(f"Saved belief log to: {belief_log_path}")

    transitions_dir = env_dir / "transitions"
    if transitions_dir.exists():
        print(f"Saved visualizations under: {transitions_dir}")
        print("Open the HTML file in a browser to export PNG/SVG for LaTeX.")
    else:  # pragma: no cover - unexpected path
        print("Finished, but no transitions directory was produced.")


if __name__ == "__main__":
    main()
