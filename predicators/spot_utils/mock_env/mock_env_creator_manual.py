"""Manual mock environment creator for testing.

This class provides functionality to:
- Create mock environments from manually collected images
- Process RGB and depth images
- Create states from multiple image views
- Support both state-first and image-first workflows
"""

from typing import Dict, Optional, Tuple, Literal, Set
import argparse
from pathlib import Path
import os
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich import print as rprint
from rich.progress import track

import numpy as np
from predicators.spot_utils.mock_env.mock_env_creator_base import MockEnvCreatorBase
from predicators.structs import Object
from predicators import utils
from predicators.settings import CFG
from predicators.spot_utils.perception.perception_structs import UnposedImageWithContext


class ManualMockEnvCreator(MockEnvCreatorBase):
    """Manual mock environment creator for testing.
    
    Directory structure for images:
        images/
        ├── state_0/
        │   ├── view1_cam1_rgb.npy
        │   ├── view1_cam1_depth.npy
        │   ├── view2_cam1_rgb.npy
        │   └── view2_cam1_depth.npy
        └── state_1/
            └── ...
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.console = Console()

    def _display_state_info(self, canonical_id: str, equivalent_ids: Set[str], state) -> None:
        """Display state information in a formatted table."""
        table = Table(title=f"State {canonical_id}", show_header=True, header_style="bold magenta")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        # Show key state information
        table.add_row("Equivalent States", ", ".join(equivalent_ids))
        table.add_row("Objects in View", ", ".join([obj.name for obj in state.objects_in_view]))
        table.add_row("Objects in Hand", ", ".join([obj.name for obj in state.objects_in_hand]))
        table.add_row("Gripper Open", "✓" if state.gripper_open else "✗")
        
        # Show key atoms
        if state_id := self.state_to_id.get(frozenset(self.id_to_state[canonical_id])):
            table.add_row("State ID", state_id)
            # Show important predicates like Inside, Holding, DrawerOpen
            atoms = self.id_to_state[canonical_id]
            important_predicates = ["Inside", "Holding", "DrawerOpen", "HandEmpty"]
            important_atoms = [str(atom) for atom in atoms 
                             if any(pred in str(atom.predicate) for pred in important_predicates)]
            if important_atoms:
                table.add_row("Key Atoms", "\n".join(important_atoms))
        
        self.console.print(table)
        self.console.print()

    def process_state_first(self, image_dir: Path) -> None:
        """Process images by iterating over states first."""
        self.console.print(Panel.fit(
            "[bold green]State-First Processing Mode[/bold green]\n\n"
            "In this mode, we'll go through each canonical state and add images to it.\n"
            "This is useful when you have multiple views per state or want to ensure complete coverage.",
            title="Welcome"
        ))
        
        # Show available canonical states
        self.console.print("\n[bold]Available Canonical States:[/bold]")
        for canonical_id, equivalent_ids in self.get_unique_states():
            # Create a table to show state info
            table = Table(title=f"State {canonical_id}", show_header=True, header_style="bold magenta")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            # Show equivalent states
            table.add_row("Equivalent States", ", ".join(equivalent_ids))
            
            # Show key predicate atoms first
            state_atoms = self.id_to_state[canonical_id]
            key_atoms = self.get_key_predicate_values(state_atoms)
            if key_atoms:
                key_atoms_str = "\n".join(str(atom) for atom in sorted(key_atoms, key=str))
                table.add_row("Key Predicates", key_atoms_str)
            
            # Show all atoms
            atoms_str = "\n".join(str(atom) for atom in sorted(state_atoms, key=str))
            table.add_row("All Atoms", atoms_str)
            
            self.console.print(table)
            self.console.print()
            
            # Track view count and collect images for this state
            view_count = 0
            collected_images = {}
            
            # Get images for this canonical state
            while True:
                img_path = Prompt.ask(
                    f"\nEnter path to image for view {view_count} of state {canonical_id} (relative to image directory, 'done'/'d' to move to next state)",
                    default="done",
                    show_default=True
                )

                if img_path.lower() in ["done", "skip", "d"]:
                    # Save all collected images for this state
                    if collected_images:
                        # First display numbered list of objects
                        self.console.print("\nAvailable objects:")
                        object_map = {}
                        for i, (obj_name, obj) in enumerate(self.objects.items(), 1):
                            object_map[str(i)] = obj
                            self.console.print(f"{i}. {obj_name}")

                        # Ask for object numbers
                        objects_in_view_str = Prompt.ask(
                            f"\nWhat objects are visible in this image? (comma-separated list of numbers, or 'none')",
                            default="none"
                        )
                        
                        # Parse objects in view
                        objects_in_view = set()
                        if objects_in_view_str.lower() != "none":
                            # Split on commas and clean up each number
                            object_nums = [num.strip() for num in objects_in_view_str.split(",")]
                            # Add each valid object to the set
                            for num in object_nums:
                                if num in object_map:
                                    objects_in_view.add(object_map[num])
                                else:
                                    self.console.print(f"[yellow]Warning: Invalid object number '{num}' - skipping[/yellow]")
                                    
                        # Show summary of collected images and objects
                        self.console.print("\nSummary for this state:")
                        self.console.print(f"\nVisible objects ({len(objects_in_view)}):")
                        for obj in sorted(objects_in_view, key=str):
                            self.console.print(f"  - {obj}")
                        self.console.print(f"Images collected ({len(collected_images)}):")
                        for camera_name in collected_images:
                            self.console.print(f"  - Camera: {camera_name}")
                            # self.console.print(f"    File: {collected_images[camera_name].rgb_path}")
                        
                        try:
                            self.save_observation(
                                state_id=canonical_id,
                                images=collected_images,
                                objects_in_view=set(),  # Empty set since we don't know objects
                                objects_in_hand=set(),  # Empty set since we don't know objects
                                gripper_open=True  # Default value
                            )
                            self.console.print(f"[green]✓ Successfully saved {len(collected_images)} images to state {canonical_id}[/green]")
                        except Exception as e:
                            self.console.print(f"[red]Error saving images: {str(e)}[/red]")
                    break
                
                # Handle both absolute and relative paths
                try:
                    # Clean up the path string - remove quotes and extra spaces
                    img_path = img_path.strip("'\"").strip()
                    
                    # Convert to Path object
                    path = Path(img_path)
                    
                    # If absolute path, use it directly
                    if path.is_absolute():
                        full_path = path
                        self.console.print(f"[yellow]Using absolute path: {path}[/yellow]")
                    else:
                        # For relative paths, make them relative to image_dir
                        full_path = image_dir / path
                        self.console.print(f"[yellow]Using relative path: {path}[/yellow]")
                    
                    if not full_path.exists():
                        self.console.print(f"[red]Image file not found at: {full_path}[/red]")
                        continue
                        
                    # Create camera name with view number and add to collected images
                    try:
                        camera_name = f"cam1.view{view_count}"
                        collected_images[camera_name] = UnposedImageWithContext(
                            rgb=self.process_rgb_image(str(full_path)),
                            depth=None,
                            camera_name=camera_name,
                            image_rot=None
                        )
                        self.console.print(f"[green]✓ Successfully processed image as view {view_count}[/green]")
                        view_count += 1
                    except Exception as e:
                        self.console.print(f"[red]Error processing image: {str(e)}[/red]")
                except Exception as e:
                    self.console.print(f"[red]Error processing path: {str(e)}[/red]")
                    continue

    def process_image_first(self, image_dir: Path) -> None:
        """Process by iterating over images first.
        
        This mode is useful when:
        - You have a directory of images to process
        - You want to assign images to states flexibly
        - You're doing exploratory mapping
        """
        self.console.print(Panel.fit(
            "[bold blue]Image-First Processing Mode[/bold blue]\n\n"
            "In this mode, we'll go through each image and assign it to a state.\n"
            "This is useful when you have a directory of images to process.",
            title="Welcome"
        ))
        
        # Get list of image files
        image_files = sorted(list(image_dir.glob("*.HEIC")))
        
        if not image_files:
            self.console.print("[red]No HEIC images found in directory![/red]")
            return
            
        # Show available states first
        self.console.print("\n[bold]Available States:[/bold]")
        for canonical_id, equivalent_ids in self.get_unique_states():
            state = self.load_state(canonical_id)
            self._display_state_info(canonical_id, equivalent_ids, state)
        
        # Process each image
        for img_file in track(image_files, description="Processing images"):
            self.console.print(Panel(
                f"[cyan]File:[/cyan] {img_file.name}\n"
                f"[cyan]Size:[/cyan] {img_file.stat().st_size / 1024:.1f} KB",
                title="Current Image"
            ))
            
            # Get state ID from user
            while True:
                state_id = Prompt.ask(
                    "Enter state ID to map this image to",
                    default="skip",
                    show_default=True,
                    help="Enter a state ID or 'skip' to skip this image"
                )
                
                if state_id.lower() == "skip":
                    self.console.print("[yellow]Skipping image...[/yellow]")
                    break
                    
                if state_id not in self.id_to_state:
                    self.console.print("[red]Invalid state ID![/red]")
                    continue
                    
                # Add image to state
                try:
                    self.add_state_from_raw_images(
                        raw_images={"cam1.seed0.rgb": (str(img_file), "rgb")},
                        state_id=state_id,
                        objects_in_view=self.load_state(state_id).objects_in_view,
                        objects_in_hand=self.load_state(state_id).objects_in_hand,
                        gripper_open=self.load_state(state_id).gripper_open
                    )
                    self.console.print(f"[green]✓ Successfully added image to state {state_id}[/green]")
                except Exception as e:
                    self.console.print(f"[red]Error adding image: {str(e)}[/red]")
                break


def main():
    """CLI interface for mapping images to states."""
    # Suppress matplotlib debug output
    os.environ['MPLBACKEND'] = 'Agg'
    
    # Get available mock environment classes
    from predicators.envs.mock_spot_env import MockSpotEnv
    import inspect
    import sys
    
    # Find all subclasses of MockSpotEnv in the module
    mock_env_classes = {}
    for name, obj in inspect.getmembers(sys.modules["predicators.envs.mock_spot_env"]):
        # if inspect.isclass(obj) and issubclass(obj, MockSpotEnv) and obj != MockSpotEnv:
        mock_env_classes[name] = obj
    
    if not mock_env_classes:
        print("No mock environment classes found!")
        return
    
    parser = argparse.ArgumentParser(description="Map images to states for mock environment")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Base directory to store environment data")
    parser.add_argument("--image_dir", type=str, required=True,
                       help="Directory containing raw images")
    parser.add_argument("--env_name", type=str, choices=list(mock_env_classes.keys()),
                       help="Name of environment class to use")
    parser.add_argument("--mode", type=str, choices=['state', 'image'], default='state',
                       help="Processing mode: 'state' to iterate over states first, 'image' to iterate over images first")
    
    args = parser.parse_args()
    
    console = Console()
    
    # Initialize CFG with default values
    utils.reset_config({
        "seed": 0,
        "approach": "oracle",
        "env": args.env_name.lower(),  # Convert to lowercase to match env name format
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "perceiver": "mock_spot_perceiver",
        "mock_env_vlm_eval_predicate": True,
        "bilevel_plan_without_sim": True,
        "horizon": 20,
    })
    
    # Create environment instance
    try:
        env_class = mock_env_classes[args.env_name]
        env = env_class(use_gui=False)  # Disable GUI to avoid warnings
        
        console.print(Panel.fit(
            "[bold green]Environment loaded successfully![/bold green]\n\n"
            f"[cyan]Name:[/cyan] {env.name}\n"
            f"[cyan]Class:[/cyan] {env_class.__name__}\n"
            f"[cyan]Objects:[/cyan] {', '.join(obj.name for obj in env.objects)}",
            title="Environment Info"
        ))
        
    except Exception as e:
        console.print(f"[red]Error loading environment: {str(e)}[/red]")
        return
    
    # Create output directory with class name suffix
    output_dir = Path(args.output_dir) / f"{env_class.__name__}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environment creator
    creator = ManualMockEnvCreator(output_dir, env=env)
    
    # First explore states and create transition graph
    console.print("\n[bold cyan]Exploring states and creating transition graph...[/bold cyan]")
    creator.explore_states(env.initial_atoms, env.objects)
    
    # Then plan and visualize transitions
    name = f'Transition Graph, {env.name.replace("_", " ").title()}'
    creator.plan_and_visualize(env.initial_atoms, env.goal_atoms, env.objects, task_name=name)
    # NOTE: Don't load existing state mapping, need unique state dict
    
    # Process based on selected mode
    image_dir = Path(args.image_dir)
    if args.mode == 'state':
        creator.process_state_first(image_dir)
    else:
        creator.process_image_first(image_dir)
    
    # Save final transitions with updated state mappings
    creator.save_transitions(save_plan=False, init_atoms=env.initial_atoms, goal_atoms=env.goal_atoms)
    console.print("\n[green]✓ Successfully saved transition system with state mapping[/green]")


if __name__ == "__main__":
    main()