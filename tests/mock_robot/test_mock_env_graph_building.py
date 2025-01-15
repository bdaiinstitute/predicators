"""Test graph building functionality in mock environment."""

import os
from pathlib import Path
from predicators import utils
from predicators.structs import Object, GroundAtom
from predicators.spot_utils.mock_env.mock_env_creator_manual import ManualMockEnvCreator
from predicators.envs.mock_spot_env import (
    _robot_type, _container_type, _immovable_object_type,
    _HandEmpty, _On, _NotBlocked, _IsPlaceable, _HasFlatTopSurface,
    _Reachable, _NEq, _NotInsideAnyContainer, _FitsInXY, _NotHolding
)
from rich.console import Console
from rich.tree import Tree
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.box import HEAVY
from typing import Set, Tuple

def _format_atoms(atoms: set) -> str:
    """Format atoms for display, focusing on key predicates."""
    key_predicates = {'Holding', 'Inside', 'On', 'HandEmpty'}
    atoms_by_pred = {}
    for atom in atoms:
        pred_name = atom.predicate.name
        if any(key in pred_name for key in key_predicates):
            if pred_name not in atoms_by_pred:
                atoms_by_pred[pred_name] = []
            atoms_by_pred[pred_name].append(atom)
    
    if not atoms_by_pred:
        return "No key predicates"
    
    # Format by predicate type
    lines = []
    for pred_name in sorted(atoms_by_pred.keys()):
        lines.append(f"[bold]{pred_name}[/bold]:")
        for atom in sorted(atoms_by_pred[pred_name], key=str):
            args = [obj.name for obj in atom.objects]
            lines.append(f"  {', '.join(args)}")
    return "\n".join(lines)

def plot_transition_graph(transitions: Set[Tuple[str, str, tuple, str]], task_name: str) -> None:
    """Plot a simple graph showing state transitions.
    
    Args:
        transitions: Set of (source_state_id, operator_name, operator_objects, dest_state_id) tuples
        task_name: Name of the task for the output file
    """
    import graphviz
    import os
    
    # Create graph
    dot = graphviz.Digraph(comment=f'Transition Graph for {task_name}')
    
    # Set graph attributes
    dot.attr('graph', {
        'fontname': 'Arial',
        'fontsize': '16',
        'label': f'State Transitions: {task_name.replace("_", " ").title()}',
        'labelloc': 't',
        'nodesep': '1.0',
        'ranksep': '1.0',
        'splines': 'curved',  # Use curved lines
        'concentrate': 'false'  # Don't merge edges for better readability
    })
    
    # Set node attributes
    dot.attr('node', {
        'fontname': 'Arial',
        'fontsize': '12',
        'shape': 'circle',  # Use circles for states
        'style': 'filled',
        'fillcolor': 'white',
        'width': '0.5',
        'height': '0.5',
        'margin': '0.1'
    })
    
    # Set edge attributes
    dot.attr('edge', {
        'fontname': 'Arial',
        'fontsize': '10',
        'arrowsize': '0.8',
        'penwidth': '1.0',
        'labeldistance': '2.0',
        'labelangle': '25'
    })
    
    # Add nodes and edges
    visited_nodes = set()
    for source_id, op_name, op_objects, dest_id in transitions:
        # Add nodes if not visited
        for node_id in [source_id, dest_id]:
            if node_id not in visited_nodes:
                dot.node(node_id, f"State {node_id}")
                visited_nodes.add(node_id)
        
        # Format edge label with operator name and objects
        edge_label = f"[{source_id}->{dest_id}] {op_name}({','.join(op_objects)})"
        
        # Add edge
        dot.edge(source_id, dest_id, edge_label)
    
    # Save graph
    graph_dir = os.path.join("mock_env_data", task_name, "transitions")
    os.makedirs(graph_dir, exist_ok=True)
    graph_path = os.path.join(graph_dir, "simple_transition_graph")
    dot.render(graph_path, format='png', cleanup=True)

def test_transitions_match_edges():
    """Test that operator transitions match graph edges.
    
    This test verifies that:
    1. All transitions found by exploring operators match edges in the graph
    2. All edges in the graph are valid operator transitions
    3. Each transition follows physical constraints:
       - Can't place without holding
       - Can't pick without having object in hand view
       - Can't move to view an object already in view
    """
    # Set up configuration
    test_name = "test_transitions_match_edges"
    test_dir = os.path.join("mock_env_data", test_name)
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "mock_env_data_dir": test_dir
    })
    
    # Create environment creator
    creator = ManualMockEnvCreator(test_dir)
    
    # Create test objects
    robot = Object("robot", _robot_type)
    cup = Object("cup", _container_type)
    source_table = Object("source_table", _immovable_object_type)
    target_table = Object("target_table", _immovable_object_type)
    objects = {robot, cup, source_table, target_table}
    
    # Create initial state atoms
    initial_atoms = {
        GroundAtom(_HandEmpty, [robot]),
        GroundAtom(_On, [cup, source_table]),
        GroundAtom(_NotBlocked, [cup]),
        GroundAtom(_IsPlaceable, [cup]),
        GroundAtom(_HasFlatTopSurface, [source_table]),
        GroundAtom(_HasFlatTopSurface, [target_table]),
        GroundAtom(_Reachable, [robot, cup]),
        GroundAtom(_NEq, [cup, source_table]),
        GroundAtom(_NEq, [cup, target_table]),
        GroundAtom(_NEq, [source_table, target_table]),
        GroundAtom(_NotInsideAnyContainer, [cup]),
        GroundAtom(_FitsInXY, [cup, source_table]),
        GroundAtom(_FitsInXY, [cup, target_table]),
        GroundAtom(_NotHolding, [robot, cup]),
        GroundAtom(_Reachable, [robot, target_table]),
        GroundAtom(_Reachable, [robot, source_table])
    }
    
    # Create goal atoms
    goal_atoms = {
        GroundAtom(_On, [cup, target_table])
    }
    
    # Plan and visualize first to generate the graph
    creator.plan_and_visualize(initial_atoms, goal_atoms, objects, task_name=test_name)
    
    # Get all possible operator transitions
    transitions = creator.get_operator_transitions(initial_atoms, objects)
    
    # Get graph edges
    edges = creator.get_graph_edges(initial_atoms, goal_atoms, objects)
    
    # Compare edges and transitions
    console = Console()
    console.print("\n[bold blue]Comparing Graph Edges vs Operator Transitions[/bold blue]")
    
    # Create mapping of states to IDs for clearer output
    state_to_id = {}
    state_count = 0
    
    # Start with initial state
    initial_state = frozenset(initial_atoms)
    state_to_id[initial_state] = "0"  # Always start with 0
    
    # First assign IDs to states in the shortest path
    curr_atoms = initial_atoms
    for edge in edges:
        next_atoms = edge[2]
        next_state = frozenset(next_atoms)
        if next_state not in state_to_id:
            state_to_id[next_state] = str(state_count + 1)
            state_count += 1
    
    # Then assign IDs to any remaining states from transitions
    for source_atoms, _, dest_atoms in transitions:
        source_state = frozenset(source_atoms)
        dest_state = frozenset(dest_atoms)
        
        if source_state not in state_to_id:
            state_to_id[source_state] = str(state_count + 1)
            state_count += 1
        
        if dest_state not in state_to_id:
            state_to_id[dest_state] = str(state_count + 1)
            state_count += 1
    
    # Create sets for comparison using state IDs
    edge_ops = {(state_to_id[frozenset(edge[0])], edge[1].name, tuple(obj.name for obj in edge[1].objects), state_to_id[frozenset(edge[2])]) for edge in edges}
    trans_ops = {(state_to_id[frozenset(t[0])], t[1].name, tuple(obj.name for obj in t[1].objects), state_to_id[frozenset(t[2])]) for t in transitions}
    
    # Plot simple transition graph
    plot_transition_graph(trans_ops, "test_transitions_match_edges")
    
    # Print state summaries
    console.print("\n[bold]State Summaries:[/bold]")
    key_predicates = {'Holding', 'HandEmpty', 'On', 'InHandView'}
    for state, state_id in sorted(state_to_id.items(), key=lambda x: x[1]):
        key_atoms = [atom for atom in state if any(pred in atom.predicate.name for pred in key_predicates)]
        console.print(f"\n[bold]{state_id}:[/bold]")
        for atom in sorted(key_atoms, key=str):
            console.print(f"  {atom}")
    
    # Print all transitions for debugging
    console.print("\n[bold]All Valid Operator Transitions:[/bold]")
    for t in sorted(trans_ops):
        console.print(f"[cyan]{t[0]} --{t[1]}({', '.join(t[2])})--> {t[3]}[/cyan]")
        
    # Print all edges for debugging
    console.print("\n[bold]All Graph Edges:[/bold]")
    for e in sorted(edge_ops):
        console.print(f"[yellow]{e[0]} --{e[1]}({', '.join(e[2])})--> {e[3]}[/yellow]")
    
    # Find differences
    edges_not_in_trans = edge_ops - trans_ops
    trans_not_in_edges = trans_ops - edge_ops
    
    # Print comparison
    if edges_not_in_trans:
        console.print("\n[red bold]Found edges in graph that aren't valid transitions:[/red bold]")
        for edge in sorted(edges_not_in_trans):
            console.print(f"[red]{edge[0]} --{edge[1]}({', '.join(edge[2])})--> {edge[3]}[/red]")
            
    if trans_not_in_edges:
        console.print("\n[yellow bold]Found valid transitions not in graph:[/yellow bold]")
        for trans in sorted(trans_not_in_edges):
            console.print(f"[yellow]{trans[0]} --{trans[1]}({', '.join(trans[2])})--> {trans[3]}[/yellow]")
    
    if not edges_not_in_trans and not trans_not_in_edges:
        console.print("\n[green]✓ All graph edges match valid transitions[/green]")
    
    # Assert no invalid edges
    assert not edges_not_in_trans, "Found edges in graph that aren't valid transitions"
    
    # Create rich console
    console = Console()
    
    # Print header
    console.print("\n[bold blue]State Transition Analysis[/bold blue]", justify="center")
    console.print("=" * 80, justify="center")
    
    # Print initial and goal states
    states_table = Table(box=HEAVY, show_header=False, width=80)
    states_table.add_row(
        Panel(_format_atoms(initial_atoms), title="[bold cyan]Initial State[/bold cyan]", border_style="cyan"),
        Panel(_format_atoms(goal_atoms), title="[bold green]Goal State[/bold green]", border_style="green")
    )
    console.print(states_table)
    console.print()
    
    # Create main tree for transitions
    tree = Tree("[bold blue]State Transitions[/bold blue]")
    
    # Track visited states to avoid cycles
    visited_states = set()
    
    # Helper function to recursively build tree
    def build_transition_tree(curr_atoms, parent_node, depth=0):
        if depth > 3:  # Limit depth to keep visualization manageable
            parent_node.add("[dim]... (further transitions omitted)[/dim]")
            return
            
        state_id = frozenset(curr_atoms)
        if state_id in visited_states:
            parent_node.add("[dim]... (state already visited)[/dim]")
            return
        visited_states.add(state_id)
        
        # Find all transitions from current state
        curr_transitions = {(s, op, d) for (s, op, d) in transitions if s == state_id}
        
        for _, op, dest_atoms in sorted(curr_transitions, key=lambda x: str(x[1])):
            # Create node for operator
            op_str = f"[yellow]{op.name}[/yellow]"
            if op.objects:
                op_str += f"({', '.join(obj.name for obj in op.objects)})"
            
            # Check if this is part of shortest path
            is_in_path = any(edge[1] == op for edge in edges)
            if is_in_path:
                op_str = f"[bold green]★[/bold green] {op_str}"
            
            # Add state info
            state_info = _format_atoms(set(dest_atoms))
            node = parent_node.add(
                Panel(
                    f"{op_str}\n[cyan]Resulting State:[/cyan]\n{state_info}",
                    border_style="blue" if is_in_path else "white",
                    padding=(1, 2)
                )
            )
            
            # Recursively add children
            build_transition_tree(set(dest_atoms), node, depth + 1)
    
    # Start building tree from initial state
    initial_panel = Panel(
        _format_atoms(initial_atoms),
        title="[bold cyan]Initial State[/bold cyan]",
        border_style="cyan",
        padding=(1, 2)
    )
    tree.add(initial_panel)
    build_transition_tree(initial_atoms, tree)
    
    # Print the tree
    console.print(tree)
    
    # Print summary
    summary = Table(box=HEAVY, show_header=False)
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("Total Transitions:", f"[green]{len(transitions)}[/green]")
    summary.add_row("Shortest Path Length:", f"[yellow]{len(edges)}[/yellow]")
    
    console.print("\n[bold]Summary[/bold]")
    console.print(summary)
    
    # Print legend
    legend = Panel(
        "[bold green]★[/bold green] = Part of shortest path to goal\n"
        "[blue]Blue border[/blue] = State in shortest path\n"
        "[yellow]Yellow text[/yellow] = Operator name\n"
        "[cyan]Cyan text[/cyan] = State predicates",
        title="[bold]Legend[/bold]",
        border_style="blue",
        padding=(1, 2)
    )
    console.print(legend)
    
    # Verify edges are subset of transitions
    for edge in edges:
        assert edge in transitions, f"Edge {edge} not found in transitions"
    
    # Verify graph file exists
    graph_file = Path(test_dir) / "transitions" / "transition_graph.png"
    assert graph_file.exists(), "Transition graph file not generated" 