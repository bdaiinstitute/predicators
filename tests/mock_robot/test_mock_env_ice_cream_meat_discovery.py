import os
import sys
import logging
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from predicators import utils
from predicators.envs.mock_spot_env import MockSpotIceCreamMeatDiscoveryEnv
from predicators.spot_utils.mock_env.mock_env_creator_base import MockEnvCreatorBase
from predicators.structs import GroundAtom

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    """Create conditional plan graph for the ice cream meat discovery task.
    
    This test creates the branching transition graph for incidental discovery planning:
    
    Scenario:
    1. Primary task: Store ice cream tub in freezer 
    2. Discovery event: When opening freezer, find unexpected meat package
    3. Information gathering: Inspect expiration date of discovered meat
    4. Conditional handling: Dispose if expired, store if fresh
    5. Complete primary task: Ice cream goes in freezer
    
    Expected branching paths:
    Path 1: No discovery - direct ice cream storage (if no discovery operator triggered)
    Path 2: Discovery + expired meat - dispose meat first, then store ice cream
    Path 3: Discovery + fresh meat - keep meat, store both items
    
    This demonstrates "determinize + replan" approach for POMDP scenarios.
    """
    # Set up configuration
    test_name = "test_mock_ice_cream_meat_discovery"
    test_dir = os.path.join("mock_env_data", test_name)
    utils.reset_config({
        "env": "mock_spot_ice_cream_meat_discovery",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "mock_env_data_dir": test_dir,
        "mock_env_use_belief_operators": True  # Enable belief operators for discovery
    })
    
    # Create environment
    env = MockSpotIceCreamMeatDiscoveryEnv()
    
    # Print environment info
    print(f"Environment: {env.get_name()}")
    print(f"Objects: {[obj.name for obj in env.objects]}")
    print(f"Operators: {[op.name for op in env.strips_operators]}")
    print(f"Initial atoms: {len(env.initial_atoms)}")
    print(f"Goal atoms: {len(env.goal_atoms)}")
    
    # Create environment creator
    creator = MockEnvCreatorBase(test_dir, env=env)
    
    # Plan and visualize transitions with branching
    name = f'Ice Cream and Meat Discovery Demo'
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
    
    print("\n🎯 Expected Branching Scenarios:")
    print("Path 1: No discovery - Robot goes straight to ice cream storage")
    print("Path 2: Discovery + expired meat - Meat found → inspected → disposed → ice cream stored") 
    print("Path 3: Discovery + fresh meat - Meat found → inspected → kept → both items stored")
    print("\n📝 Key operators for branching:")
    print("- DiscoverObjectInContainer: Triggers when meat is found")
    print("- InspectObjectExpirationExpired vs InspectObjectExpirationFresh: Creates branches")
    print("- Conditional disposal or storage based on inspection results") 