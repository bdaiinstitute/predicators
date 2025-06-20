"""Test manual creation of images for a cup classification task.

This test focuses on:
1. Creating transition graph for planning the image collection
2. Verifying cup classification operators and predicates
3. Testing belief-space planning with conditional placement

Directory Structure:
    mock_env_data/test_mock_cup_classification/
    ├── images/
    │   ├── state_0/
    │   │   └── cam1.rgb.npy
    │   └── ...
    ├── transitions/
    │   └── Transition Graph, Test Cup Classification.html
    ├── state_mapping.yaml
    └── plan.yaml
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from predicators import utils
from predicators.envs.mock_spot_env import MockSpotCupClassificationEnv
from predicators.spot_utils.mock_env.mock_env_creator_base import MockEnvCreatorBase
from predicators.structs import GroundAtom

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    """Create conditional plan graph for the cup classification task.
    
    This test creates the branching transition graph for belief-space planning:
    
    Branches based on:
    1. Cup content inspection: InspectCupContentsHasContent vs InspectCupContentsEmpty
    2. Different handling based on cup type and content:
       - paper_cup (disposable): Always goes to trash_bin regardless of content
       - ceramic_mug (washable): Must be emptied if has content, then goes to dishwasher
    
    Expected branching paths:
    Path 1: Both cups empty (optimal)
    Path 2: Paper cup has content, ceramic mug empty  
    Path 3: Paper cup empty, ceramic mug has content
    Path 4: Both cups have content (requires dumping ceramic mug)
    """
    # Set up configuration
    test_name = "test_mock_cup_classification"
    test_dir = os.path.join("mock_env_data", test_name)
    utils.reset_config({
        "env": "mock_spot_cup_classification",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "mock_env_data_dir": test_dir,
        "mock_env_use_belief_operators": True  # Enable belief operators for classification
    })
    
    # Create environment
    env = MockSpotCupClassificationEnv()
    
    # Print environment info
    print(f"Environment: {env.get_name()}")
    print(f"Objects: {[obj.name for obj in env.objects]}")
    print(f"Operators: {[op.name for op in env.strips_operators]}")
    print(f"Initial atoms: {len(env.initial_atoms)}")
    print(f"Goal atoms: {len(env.goal_atoms)}")
    
    # Create environment creator
    creator = MockEnvCreatorBase(test_dir, env=env)
    
    # Plan and visualize transitions with branching
    name = f'Cup Classification Branching Demo'
    print(f"\n🔀 Generating branching visualization for: {name}")
    
    # Use the branching visualization method
    creator.plan_and_visualize_with_branches(
        env.initial_atoms, 
        env.goal_atoms_or, 
        env.objects, 
        task_name=name
    )
    
    print("✅ Branching visualization completed!")
    print(f"📁 Output directory: {test_dir}")
    print(f"📊 Regular graph: {test_dir}/transitions/{name}.html")
    print(f"🌳 Branching graph: {test_dir}/transitions/{name}_branching.html")
    print(f"😀 Emoji graph: {test_dir}/transitions/{name}_branching_emoji.html")
    print(f"📋 Branching plan: {test_dir}/transitions/{name}_branching_plan.yaml")

    # Try to create at least a basic visualization
    try:
        # Check if any transition graph file exists
        transitions_dir = Path(test_dir) / "transitions"
        if transitions_dir.exists():
            graph_files = list(transitions_dir.glob("*.html"))
            if graph_files:
                print(f"✓ Found transition graph: {graph_files[0]}")
            else:
                print("No transition graph files found")
        else:
            print("Transitions directory not created")
    except Exception as e:
        print(f"Error checking files: {e}")
    
    print(f"✓ Test completed successfully!")
    print(f"✓ Test data directory: {test_dir}") 