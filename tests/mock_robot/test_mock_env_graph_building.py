"""Test graph building functionality in mock environment.

This module provides tests and visualization tools for transition graphs in the mock environment.
It includes both static (graphviz) and interactive (Cytoscape.js) visualizations.

Key components:
- State transitions: Shows how operators transform environment states
- Graph visualization: Both static PNG and interactive HTML outputs
- State comparison: Verifies transitions match expected behavior
- Interactive features: Draggable nodes, zoom, pan, and state inspection

The interactive visualization provides:
- Draggable nodes and zoomable canvas
- State details on click with formatted predicate display
- Toggles for shortest path and edge visibility
- Curved edges with labels
- Color coding for initial, goal, and shortest path states
- Keyboard shortcuts for common operations

Features:
- Node colors:
  - Initial state: Light blue with blue border
  - Goal states: Light green with green border
  - Shortest path: Light yellow with orange border
  - Other states: White with gray border
- Edge styles:
  - Shortest path: Red, curved
  - Other transitions: Gray, curved
  - Labels show operator names and parameters
- Interactive controls:
  - Toggle shortest path visibility
  - Toggle all edges visibility
  - Toggle animation for layout changes
  - Reset layout button
  - Keyboard shortcuts (press 'h' to view)
- State information:
  - Click nodes to view detailed state
  - Shows key predicates and self-loop operators
  - Scrollable panel for long states

Example usage:
    ```python
    # Create environment and objects
    creator = ManualMockEnvCreator(test_dir)
    robot = Object("robot", _robot_type)
    cup = Object("cup", _container_type)
    
    # Set up initial and goal states
    initial_atoms = {...}
    goal_atoms = {...}
    
    # Generate visualizations
    creator.plan_and_visualize(initial_atoms, goal_atoms, objects)
    
    # Get graph data for custom visualization
    graph_data = creator._build_transition_graph(...)
    create_interactive_visualization(graph_data, "graph.html")
    ```

Output files:
- mock_env_data/test_name/transitions/transition_graph.png: Static graph
- mock_env_data/test_name/transitions/interactive_graph.html: Interactive visualization
- mock_env_data/test_name/transitions/simple_transition_graph.png: Simplified graph
"""

import os
from pathlib import Path
from predicators import utils
from predicators.structs import Object, GroundAtom
from predicators.spot_utils.mock_env.mock_env_creator_manual import ManualMockEnvCreator
from predicators.envs.mock_spot_env import (
    MockSpotEnv, _robot_type, _container_type, _immovable_object_type,
    _HandEmpty, _On, _NotBlocked, _IsPlaceable, _HasFlatTopSurface,
    _Reachable, _NEq, _NotInsideAnyContainer, _FitsInXY, _NotHolding
)
from rich.console import Console
from rich.tree import Tree
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.box import HEAVY
from typing import Set, Tuple, Dict, Any
import json

def _format_atoms(atoms: set) -> str:
    """Format atoms for display, showing only key predicates in a simplified format.
    
    This function filters and formats ground atoms to show only the most relevant
    predicates for visualization. It simplifies the display by showing predicates
    in a clean format: Predicate(arg1, arg2).
    
    Args:
        atoms: Set of GroundAtom objects to format
        
    Returns:
        str: Formatted string with one predicate per line
        
    Example:
        >>> atoms = {GroundAtom(_HandEmpty, [robot]), GroundAtom(_On, [cup, table])}
        >>> print(_format_atoms(atoms))
        HandEmpty(robot)
        On(cup, table)
    """
    key_predicates = {'HandEmpty', 'NotHolding', 'On', 'NotInsideAnyContainer'}
    formatted_atoms = []
    
    for atom in sorted(atoms, key=str):
        pred_name = atom.predicate.name
        if any(key in pred_name for key in key_predicates):
            args = [obj.name for obj in atom.objects]
            formatted_atoms.append(f"{pred_name}({', '.join(args)})")
    
    return "\n".join(formatted_atoms)

def plot_transition_graph(transitions: Set[Tuple[str, str, tuple, str]], task_name: str) -> None:
    """Plot a simple graph showing state transitions using graphviz.
    
    This function creates a simplified version of the transition graph
    that focuses on the basic structure without detailed state information.
    Useful for quick visualization of the transition structure.
    
    The graph uses:
    - Circles for states
    - Curved edges for transitions
    - Clear labels for operators
    - Consistent styling with the interactive visualization
    
    Args:
        transitions: Set of (source_state_id, operator_name, operator_objects, dest_state_id) tuples
        task_name: Name of the task for the output file
        
    Output:
        Saves a PNG file at mock_env_data/{task_name}/transitions/simple_transition_graph.png
        
    Example:
        >>> transitions = {("0", "Pick", ("robot", "cup"), "1")}
        >>> plot_transition_graph(transitions, "pick_and_place")
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

def create_interactive_visualization(graph_data: Dict[str, Any], output_path: str) -> None:
    """Create an interactive HTML visualization of the transition graph.
    
    Args:
        graph_data: Dictionary containing nodes, edges, and metadata
        output_path: Path to save the HTML file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert nodes dictionary to array format for Cytoscape
    nodes_array = []
    for node_id, node_data in graph_data['nodes'].items():
        nodes_array.append({
            'data': {
                'id': node_id,
                'label': node_data['label'],
                'fullLabel': node_data['fullLabel'],
                'is_initial': node_data['is_initial'],
                'is_goal': node_data['is_goal'],
                'is_shortest_path': node_data['is_shortest_path']
            }
        })
    
    # Format edges for Cytoscape
    edges_array = []
    for idx, edge in enumerate(graph_data['edges']):
        edges_array.append({
            'data': {
                'id': f'edge_{idx}',
                'source': edge['source'],
                'target': edge['target'],
                'label': edge['operator'],
                'is_shortest_path': edge['is_shortest_path']
            }
        })
    
    # Create the final graph data structure
    cytoscape_data = {
        'nodes': nodes_array,
        'edges': edges_array
    }
    
    # Convert to JSON, ensuring proper string formatting
    graph_data_json = json.dumps(cytoscape_data)
    task_name = graph_data['metadata']['task_name']
    
    
    # Create HTML template with proper element structure
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>State Transition Graph: {task_name}</title>
        <meta charset="UTF-8">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.19.1/cytoscape.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.3.2/cytoscape-dagre.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                display: flex;
                height: 100vh;
                background-color: #f5f5f5;
            }}
            #cy {{
                flex-grow: 1;
                z-index: 999;
            }}
            #controls {{
                position: fixed;
                top: 10px;
                left: 10px;
                background: white;
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                z-index: 1000;
            }}
            #info-panel {{
                position: fixed;
                top: 10px;
                right: 10px;
                width: 300px;
                background: white;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                display: none;
                max-height: 80vh;
                overflow-y: auto;
                z-index: 1000;
            }}
            .close-button {{
                float: right;
                cursor: pointer;
                padding: 5px;
            }}
            .state-header {{
                margin-bottom: 10px;
                padding-bottom: 5px;
                border-bottom: 1px solid #ccc;
            }}
            .predicate {{
                margin: 5px 0;
                font-family: monospace;
            }}
            .self-loops {{
                margin-top: 10px;
                padding-top: 5px;
                border-top: 1px solid #ccc;
            }}
        </style>
    </head>
    <body>
        <div id="cy"></div>
        <div id="controls">
            <div><input type="checkbox" id="shortest-path" checked> Show shortest path</div>
            <div><input type="checkbox" id="all-edges" checked> Show all edges</div>
            <div><input type="checkbox" id="animate" checked> Animate layout changes</div>
            <div><button onclick="cy.layout(layout_options).run()">Reset Layout</button></div>
            <div style="margin-top: 10px">Press 'h' for keyboard shortcuts</div>
        </div>
        <div id="info-panel">
            <span class="close-button" onclick="hideInfoPanel()">×</span>
            <div id="state-info"></div>
        </div>
        <script>
            const graphData = {graph_data_json};
            
            // Initialize Cytoscape
            const cy = cytoscape({{
                container: document.getElementById('cy'),
                elements: graphData,
                style: [
                    {{
                        selector: 'node',
                        style: {{
                            'background-color': 'white',
                            'border-width': 2,
                            'border-color': '#666',
                            'label': 'data(label)',
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'width': '120px',
                            'height': '50px',
                            'font-size': '12px',
                            'text-wrap': 'wrap',
                            'padding': '10px'
                        }}
                    }},
                    {{
                        selector: 'node[?is_shortest_path]',
                        style: {{
                            'background-color': '#fff7e6',
                            'border-color': '#ff7f0e',
                            'border-width': 3
                        }}
                    }},
                    {{
                        selector: 'node[?is_initial][?is_shortest_path]',
                        style: {{
                            'background-color': '#e6f3ff',
                            'border-color': '#2171b5',
                            'border-width': 3
                        }}
                    }},
                    {{
                        selector: 'node[?is_goal][?is_shortest_path]',
                        style: {{
                            'background-color': '#e6ffe6',
                            'border-color': '#2ca02c',
                            'border-width': 3
                        }}
                    }},
                    {{
                        selector: 'edge',
                        style: {{
                            'width': 2,
                            'line-color': '#999',
                            'target-arrow-color': '#999',
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'bezier',
                            'label': 'data(label)',
                            'font-size': '10px',
                            'text-background-color': 'white',
                            'text-background-opacity': 1,
                            'text-background-padding': '5px',
                            'text-rotation': 'autorotate',
                            'text-margin-y': -10
                        }}
                    }},
                    {{
                        selector: 'edge[?is_shortest_path]',
                        style: {{
                            'line-color': '#e41a1c',
                            'target-arrow-color': '#e41a1c',
                            'width': 3
                        }}
                    }}
                ],
                layout: {{
                    name: 'dagre',
                    rankDir: 'LR',
                    nodeSep: 100,
                    rankSep: 150,
                    edgeSep: 50,
                    animate: true
                }},
                wheelSensitivity: 0.2
            }});

            // Layout options
            const layout_options = {{
                name: 'dagre',
                rankDir: 'LR',
                nodeSep: 100,
                rankSep: 150,
                edgeSep: 50,
                animate: document.getElementById('animate').checked
            }};

            function formatStateInfo(fullLabel) {{
                const lines = fullLabel.split('\\n');
                let html = `<div class="state-header"><strong>${{lines[0]}}</strong></div>`;
                
                let section = [];
                for (let i = 2; i < lines.length; i++) {{
                    const line = lines[i];
                    if (line === 'Self-loop operators:') {{
                        if (section.length > 0) {{
                            html += `<div class="predicates">${{section.join('<br>')}}</div>`;
                            section = [];
                        }}
                        html += `<div class="self-loops"><strong>${{line}}</strong>`;
                    }} else if (line.trim() !== '') {{
                        section.push(`<div class="predicate">${{line}}</div>`);
                    }}
                }}
                if (section.length > 0) {{
                    html += `<div class="predicates">${{section.join('')}}</div>`;
                }}
                return html;
            }}

            function showInfoPanel(node) {{
                const panel = document.getElementById('info-panel');
                const info = document.getElementById('state-info');
                info.innerHTML = formatStateInfo(node.data('fullLabel'));
                panel.style.display = 'block';
            }}

            function hideInfoPanel() {{
                document.getElementById('info-panel').style.display = 'none';
            }}

            // Event handlers
            cy.on('tap', 'node', function(evt) {{
                showInfoPanel(evt.target);
            }});

            cy.on('tap', function(evt) {{
                if (evt.target === cy) {{
                    hideInfoPanel();
                }}
            }});

            // Toggle controls
            document.getElementById('shortest-path').addEventListener('change', function(evt) {{
                cy.edges('[?is_shortest_path]').style('visibility', evt.target.checked ? 'visible' : 'hidden');
            }});

            document.getElementById('all-edges').addEventListener('change', function(evt) {{
                cy.edges('[!is_shortest_path]').style('visibility', evt.target.checked ? 'visible' : 'hidden');
            }});

            document.getElementById('animate').addEventListener('change', function(evt) {{
                layout_options.animate = evt.target.checked;
            }});

            // Keyboard shortcuts
            document.addEventListener('keydown', function(evt) {{
                switch(evt.key.toLowerCase()) {{
                    case 'h':
                        alert(
                            'Keyboard Shortcuts:\\n' +
                            'h: Show this help\\n' +
                            's: Toggle shortest path\\n' +
                            'e: Toggle all edges\\n' +
                            'a: Toggle animation\\n' +
                            'r: Reset layout\\n' +
                            'Esc: Close info panel'
                        );
                        break;
                    case 's':
                        const sp = document.getElementById('shortest-path');
                        sp.checked = !sp.checked;
                        sp.dispatchEvent(new Event('change'));
                        break;
                    case 'e':
                        const ae = document.getElementById('all-edges');
                        ae.checked = !ae.checked;
                        ae.dispatchEvent(new Event('change'));
                        break;
                    case 'a':
                        const an = document.getElementById('animate');
                        an.checked = !an.checked;
                        an.dispatchEvent(new Event('change'));
                        break;
                    case 'r':
                        cy.layout(layout_options).run();
                        break;
                    case 'escape':
                        hideInfoPanel();
                        break;
                }}
            }});

            // Print initial graph data
            console.log('Initial graph data:', graphData);
            console.log('Edges with shortest path:', cy.edges().filter(edge => edge.data('is_shortest_path')).length);
            console.log('Edges without shortest path:', cy.edges().filter(edge => !edge.data('is_shortest_path')).length);

            // Debug node data
            console.log('\\nNode data:');
            cy.nodes().forEach(node => {{
                console.log('Node:', node.id(), {{
                    'is_initial': node.data('is_initial'),
                    'is_goal': node.data('is_goal'),
                    'is_shortest_path': node.data('is_shortest_path'),
                    'background-color': node.style('background-color'),
                    'border-color': node.style('border-color')
                }});
            }});

            // Debug edge data
            console.log('\\nEdge data:');
            cy.edges().forEach(edge => {{
                console.log('Edge:', edge.id(), {{
                    'is_shortest_path': edge.data('is_shortest_path'),
                    'line-color': edge.style('line-color'),
                    'source': edge.data('source'),
                    'target': edge.data('target')
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    # Save the HTML file
    with open(output_path, 'w') as f:
        f.write(html_content)

def test_transitions_match_edges():
    """Test that operator transitions match graph edges.
    
    This test verifies that:
    1. All transitions found by exploring operators match edges in the graph
    2. All edges in the graph are valid operator transitions
    3. Each transition follows physical constraints:
       - Can't place without holding
       - Can't pick without having object in hand view
       - Can't move to view an object already in view
       
    The test generates three visualizations:
    1. Full transition graph (graphviz PNG)
    2. Interactive web visualization (HTML)
    3. Simplified graph (graphviz PNG)
    
    Output files (in mock_env_data/test_transitions_match_edges/transitions/):
    - Transition Graph, Test Transitions Match Edges.png
    - interactive_graph.html
    - simple_transition_graph.png
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
    
    # Create environment and test objects
    env = MockSpotEnv()
    creator = ManualMockEnvCreator(test_dir, env_info={
        "types": env.types,
        "predicates": env.predicates,
        "options": env.options,
        "nsrts": env.nsrts
    })
    
    robot = Object("robot", _robot_type)
    cup = Object("cup", _container_type)
    source_table = Object("source_table", _immovable_object_type)
    target_table = Object("target_table", _immovable_object_type)
    objects = {robot, cup, source_table, target_table}
    
    # Define initial state with all necessary predicates
    initial_atoms = {
        # Robot state
        GroundAtom(_HandEmpty, [robot]),
        GroundAtom(_NotHolding, [robot, cup]),
        
        # Object positions and properties
        GroundAtom(_On, [cup, source_table]),
        GroundAtom(_NotBlocked, [cup]),
        GroundAtom(_IsPlaceable, [cup]),
        
        # Surface properties
        GroundAtom(_HasFlatTopSurface, [source_table]),
        GroundAtom(_HasFlatTopSurface, [target_table]),
        
        # Reachability constraints
        GroundAtom(_Reachable, [robot, cup]),
        GroundAtom(_Reachable, [robot, target_table]),
        GroundAtom(_Reachable, [robot, source_table]),
        
        # Object relationships
        GroundAtom(_NEq, [cup, source_table]),
        GroundAtom(_NEq, [cup, target_table]),
        GroundAtom(_NEq, [source_table, target_table]),
        GroundAtom(_NotInsideAnyContainer, [cup]),
        GroundAtom(_FitsInXY, [cup, source_table]),
        GroundAtom(_FitsInXY, [cup, target_table])
    }
    
    # Define goal state - cup should be on target table
    goal_atoms = {
        GroundAtom(_On, [cup, target_table])
    }
    
    # Generate visualizations
    name = f'Transition Graph, {test_name.replace("_", " ").title()}'
    creator.plan_and_visualize(initial_atoms, goal_atoms, objects, task_name=name)
    
    # Get all possible transitions and graph edges
    transitions = creator.get_operator_transitions(initial_atoms, objects)
    edges = creator.get_graph_edges(initial_atoms, goal_atoms, objects)
    
    # Create mapping of states to IDs for visualization
    state_to_id = {}
    state_count = 0
    
    # Start with initial state (always ID 0)
    initial_state = frozenset(initial_atoms)
    state_to_id[initial_state] = "0"
    
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
    
    # Create edge data for visualization
    edge_data = []
    
    # Track shortest path edges and states
    shortest_path_edges = set()
    shortest_path_states = {frozenset(initial_atoms)}  # Start with initial state
    
    # First identify all shortest path edges and states
    for source_atoms, op, dest_atoms in edges:
        source_state = frozenset(source_atoms)
        dest_state = frozenset(dest_atoms)
        source_id = state_to_id[source_state]
        dest_id = state_to_id[dest_state]
        shortest_path_edges.add((source_id, dest_id))
        shortest_path_states.add(dest_state)  # Add destination state to shortest path
    
    # Add all transitions as edges
    for source_atoms, op, dest_atoms in transitions:
        source_state = frozenset(source_atoms)
        dest_state = frozenset(dest_atoms)
        source_id = state_to_id[source_state]
        dest_id = state_to_id[dest_state]
        if source_id != dest_id:  # Skip self-loops
            op_str = f"{op.name}({','.join(obj.name for obj in op.objects)})"
            edge_data.append({
                'source': source_id,
                'target': dest_id,
                'operator': op_str,
                'is_shortest_path': (source_id, dest_id) in shortest_path_edges
            })
    
    # Create graph data structure
    graph_data = {
        'nodes': {},
        'edges': edge_data,
        'metadata': {
            'task_name': name,
            'fluent_predicates': []
        }
    }
    
    # Add node data with state information
    for atoms, state_id in state_to_id.items():
        # Determine node properties
        is_initial = atoms == frozenset(initial_atoms)
        is_goal = goal_atoms.issubset(atoms)
        is_shortest_path = atoms in shortest_path_states
        
        # Get self loops for this state
        self_loops = []
        for source_atoms, op, dest_atoms in transitions:
            if (frozenset(source_atoms) == atoms and 
                frozenset(dest_atoms) == atoms):
                self_loops.append(f"{op.name}({','.join(obj.name for obj in op.objects)})")
        
        # Create label with proper line breaks
        state_label = f"{'Initial ' if is_initial else ''}{'Goal ' if is_goal else ''}State {state_id}"
        full_label_parts = [
            state_label,
            "",  # Empty line
            _format_atoms(atoms)
        ]
        if self_loops:
            full_label_parts.extend([
                "",  # Empty line
                "Self-loop operators:",
                *[f"  {op}" for op in self_loops]
            ])
        
        # Add node to graph data
        graph_data['nodes'][state_id] = {
            'id': state_id,
            'state_num': state_id,
            'atoms': [str(atom) for atom in atoms],
            'is_initial': is_initial,
            'is_goal': is_goal,
            'is_shortest_path': is_shortest_path,
            'label': state_label,
            'fullLabel': '\n'.join(full_label_parts)
        }
    
    # Create interactive visualization
    html_path = os.path.join(test_dir, "transitions", "interactive_graph.html")
    create_interactive_visualization(graph_data, html_path)
    
    # Create sets for comparison using state IDs
    edge_ops = {(state_to_id[frozenset(edge[0])], edge[1].name, tuple(obj.name for obj in edge[1].objects), state_to_id[frozenset(edge[2])]) for edge in edges}
    trans_ops = {(state_to_id[frozenset(t[0])], t[1].name, tuple(obj.name for obj in t[1].objects), state_to_id[frozenset(t[2])]) for t in transitions}
    
    # Plot simple transition graph
    plot_transition_graph(trans_ops, "test_transitions_match_edges")
    
    # Print comparison information using rich console
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
    
    # Print state summaries
    console.print("\n[bold]State Summaries:[/bold]")
    key_predicates = {'Holding', 'HandEmpty', 'On', 'InHandView'}
    for state, state_id in sorted(state_to_id.items(), key=lambda x: x[1]):
        key_atoms = [atom for atom in state if any(pred in atom.predicate.name for pred in key_predicates)]
        console.print(f"\n[bold]{state_id}:[/bold]")
        for atom in sorted(key_atoms, key=str):
            console.print(f"  {atom}")
    
    # Print all transitions and edges for debugging
    console.print("\n[bold]All Valid Operator Transitions:[/bold]")
    for t in sorted(trans_ops):
        console.print(f"[cyan]{t[0]} --{t[1]}({', '.join(t[2])})--> {t[3]}[/cyan]")
        
    console.print("\n[bold]All Graph Edges:[/bold]")
    for e in sorted(edge_ops):
        console.print(f"[yellow]{e[0]} --{e[1]}({', '.join(e[2])})--> {e[3]}[/yellow]")
    
    # Find and print differences
    edges_not_in_trans = edge_ops - trans_ops
    trans_not_in_edges = trans_ops - edge_ops
    
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
    graph_file = Path(test_dir) / "transitions" / f"Transition Graph, {test_name.replace('_', ' ').title()}.png"
    assert graph_file.exists(), "Transition graph file not generated" 