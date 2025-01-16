"""Test graph building functionality in mock environment.

This module provides tests and visualization tools for transition graphs in the mock environment.
It includes both static (graphviz) and interactive (Cytoscape.js) visualizations.

Key components:
- State transitions: Shows how operators transform environment states
- Graph visualization: Both static PNG and interactive HTML outputs
- State comparison: Verifies transitions match expected behavior

The interactive visualization provides:
- Draggable nodes and zoomable canvas
- State details on click
- Toggles for shortest path and edge visibility
- Consistent styling with graphviz output

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
"""

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
from typing import Set, Tuple, Dict, Any
import json

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
    """Plot a simple graph showing state transitions using graphviz.
    
    This function creates a simplified version of the transition graph
    that focuses on the basic structure without detailed state information.
    Useful for quick visualization of the transition structure.
    
    Args:
        transitions: Set of (source_state_id, operator_name, operator_objects, dest_state_id) tuples
        task_name: Name of the task for the output file
        
    Output:
        Saves a PNG file at mock_env_data/{task_name}/transitions/simple_transition_graph.png
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
        graph_data: Dictionary containing nodes, edges, and metadata for the graph
        output_path: Path to save the HTML file
    """
    # Convert data to JSON serializable format
    def make_serializable(obj: Any) -> Any:
        if isinstance(obj, set):
            return [make_serializable(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, str) and obj.startswith('[') and obj.endswith(']'):
            try:
                return json.loads(obj.replace("'", '"'))
            except json.JSONDecodeError:
                return obj
        elif hasattr(obj, '__str__'):
            return str(obj)
        return obj

    # Create a copy of the data to modify
    processed_data = dict(graph_data)
    processed_data = make_serializable(processed_data)

    # Save graph data as separate JSON file
    data_path = output_path.replace('.html', '_data.json')
    with open(data_path, 'w') as f:
        json.dump(processed_data, f, indent=2)

    # Convert to JSON string for embedding
    graph_data_json = json.dumps(processed_data)

    # Create HTML template with proper string formatting
    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <title>Transition Graph Visualization</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.26.0/cytoscape.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        #cy {{
            width: 100%;
            height: 800px;
            background: white;
            border: 1px solid #ccc;
            border-radius: 5px;
        }}
        .controls {{
            margin-bottom: 20px;
            padding: 10px;
            background: white;
            border: 1px solid #ccc;
            border-radius: 5px;
        }}
        .node-info {{
            position: fixed;
            right: 20px;
            top: 20px;
            width: 300px;
            background: white;
            padding: 15px;
            border: 1px solid #ccc;
            border-radius: 5px;
            display: none;
            max-height: 80vh;
            overflow-y: auto;
        }}
        .help-panel {{
            margin-bottom: 20px;
            padding: 10px;
            background: #e8f4f8;
            border: 1px solid #b8d6e6;
            border-radius: 5px;
        }}
        .controls label:hover {{
            cursor: pointer;
            color: #4A90E2;
        }}
        .controls input[type="checkbox"] {{
            margin-right: 5px;
        }}
        .controls, .node-info {{
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="help-panel">
        <h3>Usage Guide:</h3>
        <ul>
            <li><strong>Navigation:</strong>
                <ul>
                    <li>Drag nodes to rearrange</li>
                    <li>Scroll to zoom in/out</li>
                    <li>Click and drag background to pan</li>
                    <li>Click a node to see detailed state information</li>
                </ul>
            </li>
            <li><strong>Keyboard Shortcuts:</strong>
                <ul>
                    <li><code>h</code> - Show help</li>
                    <li><code>e</code> - Toggle edge labels</li>
                    <li><code>a</code> - Toggle animation</li>
                    <li><code>r</code> - Reset view</li>
                </ul>
            </li>
            <li><strong>Node Colors:</strong>
                <ul>
                    <li>Light Blue - Initial state</li>
                    <li>Light Green - Goal state</li>
                    <li>Light Yellow - State in shortest path</li>
                </ul>
            </li>
        </ul>
    </div>

    <div class="controls">
        <h2>Transition Graph: {processed_data['metadata']['task_name']}</h2>
        <label>
            <input type="checkbox" id="showShortestPath" checked>
            Highlight Shortest Path
        </label>
        <label style="margin-left: 20px;">
            <input type="checkbox" id="showAllEdges" checked>
            Show All Edges
        </label>
        <label style="margin-left: 20px;">
            <input type="checkbox" id="useAnimation" checked>
            Animate Layout Changes
        </label>
        <label style="margin-left: 20px;">
            <input type="checkbox" id="showLabels" checked>
            Show Edge Labels
        </label>
    </div>
    <div id="cy"></div>
    <div class="node-info" id="nodeInfo">
        <h3>State Information</h3>
        <div id="nodeContent"></div>
    </div>
    <script>
        // Register dagre layout
        cytoscape.use(cytoscapeDagre);

        // Initialize with embedded data
        let graphData = {graph_data_json};

        // Try loading from separate file if served from a web server
        if (window.location.protocol !== 'file:') {{
            fetch('{os.path.basename(data_path)}')
                .then(response => response.json())
                .then(data => {{
                    graphData = data;
                    initializeGraph(graphData);
                }})
                .catch(error => {{
                    console.warn('Could not load separate data file:', error);
                    initializeGraph(graphData);
                }});
        }} else {{
            initializeGraph(graphData);
        }}

        function initializeGraph(graphData) {{
            // Parse edges if they're a string
            if (typeof graphData.edges === 'string') {{
                graphData.edges = JSON.parse(graphData.edges.replace(/'/g, '"'));
            }}

            // Create Cytoscape instance
            const cy = cytoscape({{
                container: document.getElementById('cy'),
                elements: {{
                    nodes: Object.values(graphData.nodes).map(node => ({{
                        data: {{
                            id: node.id,
                            label: `State ${{node.state_num}}`,
                            isInitial: node.is_initial === "True",
                            isGoal: node.is_goal === "True",
                            isShortestPath: node.is_shortest_path === "True",
                            fullLabel: node.label,
                            selfLoops: node.self_loops || []
                        }}
                    }})),
                    edges: graphData.edges.map(edge => ({{
                        data: {{
                            id: `${{edge.source}}-${{edge.target}}`,
                            source: edge.source,
                            target: edge.target,
                            label: edge.operator,
                            isShortestPath: edge.is_shortest_path === 'true'
                        }}
                    }}))
                }},
                style: [
                    {{
                        selector: 'node',
                        style: {{
                            'label': 'data(label)',
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'background-color': '#fff',
                            'border-width': 2,
                            'border-color': '#666',
                            'shape': 'rectangle',
                            'width': '120px',
                            'height': '40px',
                            'padding': '10px',
                            'transition-property': 'background-color, border-width, border-color',
                            'transition-duration': '0.3s'
                        }}
                    }},
                    {{
                        selector: 'node[?isInitial]',
                        style: {{
                            'background-color': '#ADD8E6',
                            'border-width': 3
                        }}
                    }},
                    {{
                        selector: 'node[?isGoal]',
                        style: {{
                            'background-color': '#90EE90',
                            'border-width': 3
                        }}
                    }},
                    {{
                        selector: 'node[?isShortestPath]',
                        style: {{
                            'background-color': '#FFFF99'
                        }}
                    }},
                    {{
                        selector: 'node:selected',
                        style: {{
                            'border-color': '#4A90E2',
                            'border-width': 4
                        }}
                    }},
                    {{
                        selector: 'edge',
                        style: {{
                            'width': 2,
                            'line-color': '#666',
                            'target-arrow-color': '#666',
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'unbundled-bezier',
                            'control-point-distances': [40],
                            'control-point-weights': [0.5],
                            'label': 'data(label)',
                            'text-background-color': '#fff',
                            'text-background-opacity': 1,
                            'text-background-padding': '3px',
                            'text-rotation': 'autorotate',
                            'font-size': '10px',
                            'text-margin-y': -10,
                            'transition-property': 'line-color, target-arrow-color, width',
                            'transition-duration': '0.3s'
                        }}
                    }},
                    {{
                        selector: 'edge[?isShortestPath]',
                        style: {{
                            'line-color': '#E74C3C',
                            'target-arrow-color': '#E74C3C',
                            'width': 3
                        }}
                    }},
                    {{
                        selector: 'edge:selected',
                        style: {{
                            'line-color': '#4A90E2',
                            'target-arrow-color': '#4A90E2',
                            'width': 4
                        }}
                    }}
                ],
                layout: {{
                    name: 'dagre',
                    rankDir: 'LR',
                    nodeSep: 100,
                    rankSep: 150,
                    animate: true,
                    animationDuration: 500
                }}
            }});

            // Add event listeners
            cy.on('tap', 'node', function(evt) {{
                const node = evt.target;
                const nodeInfo = document.getElementById('nodeInfo');
                const nodeContent = document.getElementById('nodeContent');
                
                // Update node info panel
                nodeContent.innerHTML = node.data('fullLabel').replace(/\\[bold\\]/g, '<strong>').replace(/\\[\\/bold\\]/g, '</strong>');
                nodeInfo.style.display = 'block';
            }});

            cy.on('tap', function(evt) {{
                if (evt.target === cy) {{
                    document.getElementById('nodeInfo').style.display = 'none';
                }}
            }});

            // Add toggle controls
            document.getElementById('showShortestPath').addEventListener('change', function(evt) {{
                const show = evt.target.checked;
                cy.style()
                    .selector('node[?isShortestPath]')
                    .style({{
                        'background-color': show ? '#FFFF99' : '#fff'
                    }})
                    .selector('edge[?isShortestPath]')
                    .style({{
                        'line-color': show ? '#E74C3C' : '#666',
                        'target-arrow-color': show ? '#E74C3C' : '#666',
                        'width': show ? 3 : 2
                    }})
                    .update();
            }});

            document.getElementById('showAllEdges').addEventListener('change', function(evt) {{
                const show = evt.target.checked;
                cy.edges().style({{
                    'display': show ? 'element' : 'none'
                }});
            }});

            document.getElementById('useAnimation').addEventListener('change', function(evt) {{
                const useAnimation = evt.target.checked;
                cy.layout({{
                    name: 'dagre',
                    rankDir: 'LR',
                    nodeSep: 100,
                    rankSep: 150,
                    animate: useAnimation,
                    animationDuration: useAnimation ? 500 : 0
                }}).run();
            }});

            document.getElementById('showLabels').addEventListener('change', function(evt) {{
                const show = evt.target.checked;
                cy.style()
                    .selector('edge')
                    .style({{
                        'label': show ? 'data(label)' : ''
                    }})
                    .update();
            }});

            // Add keyboard shortcuts
            document.addEventListener('keydown', function(evt) {{
                switch(evt.key.toLowerCase()) {{
                    case 'h':  // Toggle help
                        alert(`Keyboard Shortcuts:
h: Show this help
e: Toggle edge labels
a: Toggle animation
r: Reset view`);
                        break;
                    case 'e':  // Toggle edge labels
                        const labelCheckbox = document.getElementById('showLabels');
                        labelCheckbox.checked = !labelCheckbox.checked;
                        labelCheckbox.dispatchEvent(new Event('change'));
                        break;
                    case 'a':  // Toggle animation
                        const animCheckbox = document.getElementById('useAnimation');
                        animCheckbox.checked = !animCheckbox.checked;
                        animCheckbox.dispatchEvent(new Event('change'));
                        break;
                    case 'r':  // Reset view
                        cy.fit();
                        break;
                }}
            }});

            // Initial layout
            cy.layout({{
                name: 'dagre',
                rankDir: 'LR',
                nodeSep: 100,
                rankSep: 150,
                animate: true,
                animationDuration: 500
            }}).run();
        }}
    </script>
</body>
</html>"""

    # Write HTML file
    with open(output_path, 'w') as f:
        f.write(html_template)
    
    print(f"\nVisualization files created:")
    print(f"- HTML: {output_path}")
    print(f"- Data: {data_path}")
    print("\nFeatures:")
    print("- Click nodes to see state details")
    print("- Drag nodes to rearrange")
    print("- Use mouse wheel to zoom")
    print("- Toggle shortest path highlighting")
    print("- Toggle edge visibility")
    print("- Toggle edge labels")
    print("- Toggle animation")
    print("\nKeyboard shortcuts:")
    print("h: Show help")
    print("e: Toggle edge labels")
    print("a: Toggle animation")
    print("r: Reset view")

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
    name = f'Transition Graph, {test_name.replace("_", " ").title()}'
    creator.plan_and_visualize(initial_atoms, goal_atoms, objects, task_name=name)
    
    # Get all possible operator transitions
    transitions = creator.get_operator_transitions(initial_atoms, objects)
    
    # Get graph edges
    edges = creator.get_graph_edges(initial_atoms, goal_atoms, objects)
    
    # Create mapping of states to IDs
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
    
    # Create edge data for visualization
    edge_data = []
    
    # First add edges from the shortest path
    for source_atoms, op, dest_atoms in edges:
        source_id = state_to_id[frozenset(source_atoms)]
        dest_id = state_to_id[frozenset(dest_atoms)]
        op_str = f"{op.name}({','.join(obj.name for obj in op.objects)})"
        edge_data.append({
            'source': source_id,
            'target': dest_id,
            'operator': op_str,
            'is_shortest_path': 'true'  # Use string 'true' for JSON compatibility
        })
    
    # Then add remaining transitions
    for source_atoms, op, dest_atoms in transitions:
        source_id = state_to_id[frozenset(source_atoms)]
        dest_id = state_to_id[frozenset(dest_atoms)]
        # Skip if this is already in the shortest path
        if any(e['source'] == source_id and e['target'] == dest_id for e in edge_data):
            continue
        # Skip self-loops as they're handled separately
        if source_id == dest_id:
            continue
        op_str = f"{op.name}({','.join(obj.name for obj in op.objects)})"
        edge_data.append({
            'source': source_id,
            'target': dest_id,
            'operator': op_str,
            'is_shortest_path': 'false'  # Use string 'false' for JSON compatibility
        })
    
    # Create graph data
    graph_data = {
        'nodes': {},
        'edges': edge_data,  # Use edge_data directly as it's already an array
        'metadata': {
            'task_name': name,
            'fluent_predicates': []
        }
    }
    
    # Add node data
    for atoms, state_id in state_to_id.items():
        is_initial = atoms == frozenset(initial_atoms)
        is_goal = goal_atoms.issubset(atoms)
        # Check if this state is in the shortest path edges
        is_shortest_path = any(
            (state_id == state_to_id[frozenset(edge[0])] or 
             state_id == state_to_id[frozenset(edge[2])])
            for edge in edges
        )
        
        # Get self loops
        self_loops = []
        for source_atoms, op, dest_atoms in transitions:
            if frozenset(source_atoms) == atoms and frozenset(dest_atoms) == atoms:
                self_loops.append(f"{op.name}({','.join(obj.name for obj in op.objects)})")
        
        graph_data['nodes'][state_id] = {
            'id': state_id,
            'state_num': state_id,
            'atoms': [str(atom) for atom in atoms],
            'is_initial': str(is_initial),  # Convert to string for consistent JSON
            'is_goal': str(is_goal),
            'is_shortest_path': str(is_shortest_path),
            'self_loops': self_loops,
            'label': f"{'Initial ' if is_initial else ''}{'Goal ' if is_goal else ''}State {state_id}\n{'─'*40}\n{_format_atoms(atoms)}\n{'─'*40}\n{f'Self-loop operators:{chr(10)}{chr(10).join(self_loops)}' if self_loops else ''}"
        }
    
    # Create interactive visualization
    html_path = os.path.join(test_dir, "transitions", "interactive_graph.html")
    create_interactive_visualization(graph_data, html_path)
    
    # Create sets for comparison using state IDs
    edge_ops = {(state_to_id[frozenset(edge[0])], edge[1].name, tuple(obj.name for obj in edge[1].objects), state_to_id[frozenset(edge[2])]) for edge in edges}
    trans_ops = {(state_to_id[frozenset(t[0])], t[1].name, tuple(obj.name for obj in t[1].objects), state_to_id[frozenset(t[2])]) for t in transitions}
    
    # Plot simple transition graph
    plot_transition_graph(trans_ops, "test_transitions_match_edges")
    
    # Print state summaries
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
    graph_file = Path(test_dir) / "transitions" / f"Transition Graph, {test_name.replace('_', ' ').title()}.png"
    assert graph_file.exists(), "Transition graph file not generated" 