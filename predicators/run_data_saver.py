"""Module for saving comprehensive run data including task plans, states, VLM atoms, and execution details."""

import json
import pickle as pkl
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Sequence
from collections import defaultdict
import logging
import yaml

from predicators.structs import State, Action, Task, LowLevelTrajectory, AugmentedState
from predicators import utils


class RunDataSaver:
    """Saves comprehensive run data for each episode/task execution."""
    
    def __init__(self, base_output_dir: str = "run_data"):
        """Initialize the run data saver.
        
        Args:
            base_output_dir: Base directory to save all run data
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Current run data
        self.current_run_data: Dict[str, Any] = {}
        self.current_run_id: Optional[str] = None
        
        # Planned actions storage
        self.planned_actions: List[Dict[str, Any]] = []
        self.executed_actions: List[Dict[str, Any]] = []
        
        # State and VLM data storage
        self.states_history: List[Dict[str, Any]] = []
        self.vlm_atoms_history: List[Dict[str, Any]] = []
        
    def start_new_run(self, task: Task, run_config: Dict[str, Any]) -> str:
        """Start tracking a new run/episode.
        
        Args:
            task: The task being executed
            run_config: Configuration for this run
            
        Returns:
            Unique run ID
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        task_idx = getattr(task, 'task_idx', 0)
        self.current_run_id = f"run_{timestamp}_{task_idx}"
        
        self.current_run_data = {
            "run_id": self.current_run_id,
            "timestamp": timestamp,
            "task": self._serialize_task(task),
            "config": run_config,
            "git_commit_hash": utils.get_git_commit_hash(),
            "planned_actions": [],
            "executed_actions": [],
            "states_history": [],
            "vlm_atoms_history": [],
            "final_metrics": {},
            "execution_summary": {}
        }
        
        # Reset per-run storage
        self.planned_actions = []
        self.executed_actions = []
        self.states_history = []
        self.vlm_atoms_history = []
        
        logging.info(f"Started tracking run: {self.current_run_id}")
        return self.current_run_id
    
    def save_planned_actions(self, planned_actions: List[Any]) -> None:
        """Save the planned actions from the approach.
        
        Args:
            planned_actions: List of planned actions/options
        """
        if not self.current_run_id:
            logging.warning("No active run to save planned actions to")
            return
            
        serialized_actions = []
        for action in planned_actions:
            if hasattr(action, 'name') and hasattr(action, 'objects'):
                # This is likely an _Option object
                serialized_actions.append({
                    "type": "option",
                    "name": action.name,
                    "objects": [str(obj) for obj in action.objects],
                    "str_repr": str(action)
                })
            else:
                # Generic action
                serialized_actions.append({
                    "type": "action",
                    "str_repr": str(action),
                    "details": getattr(action, '__dict__', {})
                })
        
        self.planned_actions = serialized_actions
        self.current_run_data["planned_actions"] = serialized_actions
        
        logging.info(f"Saved {len(serialized_actions)} planned actions for run {self.current_run_id}")
    
    def save_executed_action(self, action: Optional[Action], step_num: int) -> None:
        """Save an executed action.
        
        Args:
            action: The executed action (None if no action taken)
            step_num: Step number in the episode
        """
        if not self.current_run_id:
            return
            
        if action is None:
            action_data = {
                "step": step_num,
                "action": None,
                "action_type": "wait",
                "details": "No actions available. Wait action taken."
            }
        else:
            action_data = {
                "step": step_num,
                "action": str(action),
                "action_type": type(action).__name__,
                "details": {}
            }
            
            # Extract extra info if available
            if hasattr(action, 'extra_info') and action.extra_info:
                if "operator_name" in action.extra_info:
                    action_data["operator_name"] = action.extra_info["operator_name"]
                    objects = action.extra_info.get("objects", [])
                    action_data["objects"] = [obj.name for obj in objects]
                    action_data["formatted"] = f"{action.extra_info['operator_name']}({', '.join([obj.name for obj in objects])})"
                action_data["extra_info"] = action.extra_info
        
        self.executed_actions.append(action_data)
        self.current_run_data["executed_actions"] = self.executed_actions
    
    def save_state_and_vlm_data(self, state: State, step_num: int) -> None:
        """Save state and VLM atom data.
        
        Args:
            state: Current state
            step_num: Step number in the episode
        """
        if not self.current_run_id:
            return
            
        state_data = {
            "step": step_num,
            "timestamp": time.time(),
            "objects": [str(obj) for obj in state.data.keys()],
            "num_objects": len(state.data)
        }
        
        # Extract VLM atoms if this is an AugmentedState
        if isinstance(state, AugmentedState):
            vlm_data = {
                "step": step_num,
                "timestamp": time.time(),
                "vlm_atoms": {},
                "true_vlm_atoms": {},
                "vlm_predicates": []
            }
            
            if state.vlm_atom_dict:
                # Save all VLM atoms
                vlm_data["vlm_atoms"] = {str(atom): value for atom, value in state.vlm_atom_dict.items()}
                
                # Save only True VLM atoms
                true_atoms = {str(atom): value for atom, value in state.vlm_atom_dict.items() if value}
                vlm_data["true_vlm_atoms"] = true_atoms
                
                state_data["vlm_atoms_count"] = len(state.vlm_atom_dict)
                state_data["true_vlm_atoms_count"] = len(true_atoms)
            
            if state.vlm_predicates:
                vlm_data["vlm_predicates"] = [str(pred) for pred in state.vlm_predicates]
                
            if state.non_vlm_atom_dict:
                vlm_data["non_vlm_atoms"] = {str(atom): value for atom, value in state.non_vlm_atom_dict.items()}
                
            self.vlm_atoms_history.append(vlm_data)
            self.current_run_data["vlm_atoms_history"] = self.vlm_atoms_history
        
        self.states_history.append(state_data)
        self.current_run_data["states_history"] = self.states_history
    
    def save_final_metrics(self, metrics: Dict[str, Any], solved: bool, trajectory: Optional[LowLevelTrajectory] = None) -> None:
        """Save final metrics and execution summary.
        
        Args:
            metrics: Final metrics from the run
            solved: Whether the task was solved
            trajectory: Optional trajectory data
        """
        if not self.current_run_id:
            return
            
        execution_summary = {
            "solved": solved,
            "num_planned_actions": len(self.planned_actions),
            "num_executed_actions": len(self.executed_actions),
            "num_states": len(self.states_history),
            "num_vlm_evaluations": len(self.vlm_atoms_history)
        }
        
        if trajectory:
            execution_summary.update({
                "trajectory_length": len(trajectory.actions),
                "num_states_in_trajectory": len(trajectory.states)
            })
        
        self.current_run_data["final_metrics"] = dict(metrics)
        self.current_run_data["execution_summary"] = execution_summary
    
    def finalize_and_save_run(self) -> Optional[str]:
        """Finalize and save the current run data to disk.
        
        Returns:
            Path to saved file if successful, None otherwise
        """
        if not self.current_run_id or not self.current_run_data:
            logging.warning("No active run to save")
            return None
        
        # Create run-specific directory
        run_dir = self.base_output_dir / self.current_run_id
        run_dir.mkdir(exist_ok=True)
        
        # Save as both pickle and JSON/YAML for different use cases
        pickle_file = run_dir / "run_data.pkl"
        yaml_file = run_dir / "run_data.yaml"
        json_file = run_dir / "run_data.json"
        
        # Save pickle (full data with all Python objects)
        with open(pickle_file, "wb") as f:
            pkl.dump(self.current_run_data, f)
        
        # Save YAML (human-readable)
        yaml_friendly_data = self._make_yaml_friendly(self.current_run_data)
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_friendly_data, f, default_flow_style=False, indent=2)
        
        # Save JSON (for programmatic access)
        json_friendly_data = self._make_json_friendly(self.current_run_data)
        with open(json_file, "w") as f:
            json.dump(json_friendly_data, f, indent=2)
        
        # Save summary text file
        summary_file = run_dir / "summary.txt"
        self._save_human_readable_summary(summary_file)
        
        logging.info(f"Saved run data to {run_dir}")
        logging.info(f"Files: {pickle_file.name}, {yaml_file.name}, {json_file.name}, {summary_file.name}")
        
        # Reset for next run
        self.current_run_id = None
        self.current_run_data = {}
        
        return str(run_dir)
    
    def _serialize_task(self, task: Task) -> Dict[str, Any]:
        """Serialize a task to a dictionary."""
        return {
            "task_idx": getattr(task, 'task_idx', 0),
            "init_goal": str(task.goal),
            "init_goal_atoms": [str(atom) for atom in task.goal],
            "str_repr": str(task)
        }
    
    def _make_yaml_friendly(self, obj: Any) -> Any:
        """Convert objects to YAML-friendly format."""
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        if isinstance(obj, (list, tuple)):
            return [self._make_yaml_friendly(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self._make_yaml_friendly(v) for k, v in obj.items()}
        if isinstance(obj, defaultdict):
            return {k: self._make_yaml_friendly(v) for k, v in dict(obj).items()}
        # Handle special objects
        if hasattr(obj, '__dict__'):
            clean_dict = {}
            for k, v in vars(obj).items():
                if not callable(v) and not k.startswith('_'):
                    try:
                        clean_dict[k] = self._make_yaml_friendly(v)
                    except:
                        pass
            return clean_dict
        return str(obj)
    
    def _make_json_friendly(self, obj: Any) -> Any:
        """Convert objects to JSON-friendly format."""
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        if isinstance(obj, (list, tuple)):
            return [self._make_json_friendly(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self._make_json_friendly(v) for k, v in obj.items()}
        if isinstance(obj, defaultdict):
            return {k: self._make_json_friendly(v) for k, v in dict(obj).items()}
        return str(obj)
    
    def _save_human_readable_summary(self, filepath: Path) -> None:
        """Save a human-readable summary of the run."""
        with open(filepath, "w") as f:
            f.write(f"Run Summary: {self.current_run_id}\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic info
            f.write(f"Timestamp: {self.current_run_data.get('timestamp', 'N/A')}\n")
            f.write(f"Git Commit: {self.current_run_data.get('git_commit_hash', 'N/A')}\n\n")
            
            # Task info
            task_data = self.current_run_data.get('task', {})
            f.write(f"Task Index: {task_data.get('task_idx', 'N/A')}\n")
            f.write(f"Goal: {task_data.get('init_goal', 'N/A')}\n\n")
            
            # Execution summary
            summary = self.current_run_data.get('execution_summary', {})
            f.write("Execution Summary:\n")
            f.write(f"  Solved: {summary.get('solved', 'N/A')}\n")
            f.write(f"  Planned Actions: {summary.get('num_planned_actions', 0)}\n")
            f.write(f"  Executed Actions: {summary.get('num_executed_actions', 0)}\n")
            f.write(f"  State Evaluations: {summary.get('num_states', 0)}\n")
            f.write(f"  VLM Evaluations: {summary.get('num_vlm_evaluations', 0)}\n\n")
            
            # Planned actions
            f.write("Planned Actions:\n")
            for i, action in enumerate(self.current_run_data.get('planned_actions', [])):
                f.write(f"  {i+1}. {action.get('formatted', action.get('str_repr', str(action)))}\n")
            f.write("\n")
            
            # Executed actions
            f.write("Executed Actions:\n")
            for action in self.current_run_data.get('executed_actions', []):
                step = action.get('step', '?')
                if action.get('action') is None:
                    f.write(f"  Step {step}: No actions available. Wait action taken.\n")
                else:
                    formatted = action.get('formatted', action.get('action', 'Unknown'))
                    f.write(f"  Step {step}: {formatted}\n")
            f.write("\n")
            
            # Final metrics (subset)
            metrics = self.current_run_data.get('final_metrics', {})
            if metrics:
                f.write("Key Metrics:\n")
                for key in ['num_solved', 'num_total', 'avg_suc_time', 'total_steps']:
                    if key in metrics:
                        value = metrics[key]
                        if isinstance(value, float):
                            f.write(f"  {key}: {value:.5f}\n")
                        else:
                            f.write(f"  {key}: {value}\n")


# Global instance for easy access
_global_saver: Optional[RunDataSaver] = None


def get_run_data_saver() -> RunDataSaver:
    """Get the global run data saver instance."""
    global _global_saver
    if _global_saver is None:
        # Create output directory based on current config
        try:
            from predicators.settings import CFG
            output_dir = f"{CFG.results_dir}/run_data"
        except:
            output_dir = "run_data"
        _global_saver = RunDataSaver(output_dir)
    return _global_saver


def initialize_run_data_saver(base_output_dir: Optional[str] = None) -> RunDataSaver:
    """Initialize the global run data saver."""
    global _global_saver
    if base_output_dir is None:
        try:
            from predicators.settings import CFG
            base_output_dir = f"{CFG.results_dir}/run_data"
        except:
            base_output_dir = "run_data"
    _global_saver = RunDataSaver(base_output_dir)
    return _global_saver 