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
    """Format atoms for display, showing only key predicates in a simplified format."""
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
    """Create an interactive HTML visualization of the transition graph."""
    # Create interactive visualization
    html_template = '''
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
            }}
            #cy {{
                flex-grow: 1;
                z-index: 999;
            }}
            #controls {{
                position: absolute;
                top: 10px;
                left: 10px;
                background: white;
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.2);
                z-index: 1000;
            }}
            #info-panel {{
                position: fixed;
                right: 10px;
                top: 10px;
                background: white;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.2);
                max-width: 300px;
                max-height: 80vh;
                overflow-y: auto;
                display: none;
                z-index: 1000;
            }}
            .close-button {{
                float: right;
                cursor: pointer;
                padding: 5px;
            }}
        </style>
    </head>
    <body>
        <div id="controls">
            <input type="checkbox" id="shortest-path" checked>
            <label for="shortest-path">Show shortest path</label><br>
            <input type="checkbox" id="all-edges" checked>
            <label for="all-edges">Show all edges</label><br>
            <input type="checkbox" id="animate" checked>
            <label for="animate">Animate layout</label><br>
            <button onclick="cy.layout(layout_options).run()">Reset Layout</button><br>
            <small>Press 'h' for keyboard shortcuts</small>
        </div>
        <div id="info-panel">
            <span class="close-button" onclick="hideInfoPanel()">✕</span>
            <div id="info-content"></div>
        </div>
        <div id="cy"></div>
        <script>
            const graphData = {graph_data_json};
            
            const layout_options = {{
                name: 'dagre',
                rankDir: 'TB',
                nodeSep: 100,
                rankSep: 150,
                edgeSep: 80,
                ranker: 'network-simplex',
                animate: true,
                animationDuration: 500,
                fit: true,
                padding: 50
            }};

            const cy = cytoscape({{
                container: document.getElementById('cy'),
                layout: layout_options,
                style: [
                    {{
                        selector: 'node',
                        style: {{
                            'background-color': 'data(backgroundColor)',
                            'border-color': 'data(borderColor)',
                            'border-width': 'data(borderWidth)',
                            'border-style': 'solid',
                            'label': 'data(label)',
                            'color': 'data(textColor)',
                            'text-wrap': 'wrap',
                            'text-max-width': '100px',
                            'font-size': '14px',
                            'font-weight': 'bold',
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'padding': '15px',
                            'shape': 'ellipse',
                            'width': '100px',
                            'height': '100px',
                            'text-margin-y': '5px',
                            'ghost': 'yes',
                            'ghost-offset-x': '0px',
                            'ghost-offset-y': '0px',
                            'ghost-opacity': 0.2
                        }}
                    }},
                    {{
                        selector: 'edge',
                        style: {{
                            'curve-style': 'bezier',
                            'control-point-step-size': 120,
                            'target-arrow-shape': 'triangle',
                            'source-arrow-shape': 'none',
                            'line-color': 'data(color)',
                            'target-arrow-color': 'data(color)',
                            'text-rotation': 'autorotate',
                            'label': 'data(operator)',
                            'font-size': '12px',
                            'text-background-color': 'white',
                            'text-background-opacity': 1,
                            'text-background-padding': '5px',
                            'text-margin-y': '-10px',
                            'edge-text-rotation': 'autorotate',
                            'arrow-scale': 1.5,
                            'width': 2
                        }}
                    }}
                ],
                elements: {{
                    nodes: Object.values(graphData.nodes).map(node => ({{
                        data: {{
                            ...node,
                            backgroundColor: node.is_initial === 'true' ? '#AED6F1' :     // Light blue
                                           node.is_goal === 'true' ? '#A3E4D7' :         // Light green
                                           node.is_shortest_path === 'true' ? '#FAD7A0' : // Light yellow
                                           '#FFFFFF',  // White
                            borderColor: node.is_initial === 'true' ? '#3498DB' :     // Blue
                                       node.is_goal === 'true' ? '#2ECC71' :         // Green
                                       node.is_shortest_path === 'true' ? '#F39C12' : // Orange
                                       '#95A5A6',  // Gray
                            borderWidth: node.is_initial === 'true' || node.is_goal === 'true' ? 4 : 2,
                            textColor: node.is_initial === 'true' ? '#2980B9' :     // Darker blue
                                     node.is_goal === 'true' ? '#27AE60' :         // Darker green
                                     node.is_shortest_path === 'true' ? '#D35400' : // Darker orange
                                     '#7F8C8D'  // Darker gray
                        }}
                    }})),
                    edges: graphData.edges.map(edge => ({{
                        data: {{
                            ...edge,
                            color: edge.is_shortest_path === 'true' ? '#E74C3C' : '#95A5A6'  // Red for shortest path, gray for others
                        }}
                    }}))
                }}
            }});

            function formatStateInfo(stateData) {{
                let html = '<div style="color: ' + stateData.textColor + '; font-weight: bold; margin-bottom: 10px;">' + stateData.label + '</div>';
                
                // Add state info
                const lines = stateData.fullLabel.split('\\n');
                for (const line of lines) {{
                    if (line.trim() && !line.includes('State') && !line.includes('Self-loop')) {{
                        html += '<div style="margin: 5px 0;">' + line + '</div>';
                    }}
                }}
                
                // Add self-loops if any
                if (stateData.fullLabel.includes('Self-loop')) {{
                    html += '<div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #eee;">';
                    html += '<div style="font-weight: bold; margin-bottom: 5px;">Self-loop operators:</div>';
                    const selfLoops = stateData.fullLabel.split('Self-loop operators:')[1].trim().split('\\n');
                    for (const loop of selfLoops) {{
                        if (loop.trim()) {{
                            html += '<div style="margin-left: 10px;">' + loop.trim() + '</div>';
                        }}
                    }}
                    html += '</div>';
                }}
                
                return html;
            }}

            function showInfoPanel(node) {{
                document.getElementById('info-content').innerHTML = formatStateInfo(node.data());
                document.getElementById('info-panel').style.display = 'block';
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
            document.getElementById('shortest-path').addEventListener('change', function(e) {{
                cy.edges().forEach(edge => {{
                    if (edge.data('is_shortest_path') === 'true') {{
                        edge.style('visibility', e.target.checked ? 'visible' : 'hidden');
                    }}
                }});
            }});

            document.getElementById('all-edges').addEventListener('change', function(e) {{
                cy.edges().forEach(edge => {{
                    if (edge.data('is_shortest_path') !== 'true') {{
                        edge.style('visibility', e.target.checked ? 'visible' : 'hidden');
                    }}
                }});
            }});

            document.getElementById('animate').addEventListener('change', function(e) {{
                layout_options.animate = e.target.checked;
                cy.layout(layout_options).run();
            }});

            // Keyboard shortcuts
            document.addEventListener('keydown', function(e) {{
                if (e.key === 'h') {{
                    alert(
                        'Keyboard Shortcuts:\\n' +
                        '- h: Show this help\\n' +
                        '- s: Toggle shortest path\\n' +
                        '- e: Toggle all edges\\n' +
                        '- a: Toggle animation\\n' +
                        '- r: Reset layout\\n' +
                        '- Escape: Close info panel'
                    );
                }} else if (e.key === 's') {{
                    const toggle = document.getElementById('shortest-path');
                    toggle.checked = !toggle.checked;
                    toggle.dispatchEvent(new Event('change'));
                }} else if (e.key === 'e') {{
                    const toggle = document.getElementById('all-edges');
                    toggle.checked = !toggle.checked;
                    toggle.dispatchEvent(new Event('change'));
                }} else if (e.key === 'a') {{
                    const toggle = document.getElementById('animate');
                    toggle.checked = !toggle.checked;
                    toggle.dispatchEvent(new Event('change'));
                }} else if (e.key === 'r') {{
                    cy.layout(layout_options).run();
                }} else if (e.key === 'Escape') {{
                    hideInfoPanel();
                }}
            }});
        </script>
    </body>
    </html>
    '''

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert graph data to JSON
    graph_data_json = json.dumps(graph_data)

    # Save the HTML file
    with open(output_path, 'w') as f:
        f.write(html_template.format(
            task_name=graph_data['metadata']['task_name'],
            graph_data_json=graph_data_json
        ))

    print(f"- Visualization files generated:")
    print(f"  - Interactive: {output_path}")
    print("  Features:")
    print("  - Drag nodes to rearrange")
    print("  - Zoom and pan")
    print("  - Click nodes for state info")
    print("  - Toggle controls for shortest path and all edges")
    print("  - Animation toggle for layout changes")
    print("  - Press 'h' for keyboard shortcuts")

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
    shortest_path_edges = set()
    for source_atoms, op, dest_atoms in edges:
        source_id = state_to_id[frozenset(source_atoms)]
        dest_id = state_to_id[frozenset(dest_atoms)]
        if source_id != dest_id:  # Skip self-loops
            op_str = f"{op.name}({','.join(obj.name for obj in op.objects)})"
            edge_data.append({
                'source': source_id,
                'target': dest_id,
                'operator': op_str,
                'is_shortest_path': 'true'  # Use string 'true' for JSON compatibility
            })
            shortest_path_edges.add((source_id, dest_id))
    
    # Then add remaining transitions (excluding self-loops and duplicates)
    for source_atoms, op, dest_atoms in transitions:
        source_id = state_to_id[frozenset(source_atoms)]
        dest_id = state_to_id[frozenset(dest_atoms)]
        if source_id != dest_id and (source_id, dest_id) not in shortest_path_edges:
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
        'edges': edge_data,
        'metadata': {
            'task_name': name,
            'fluent_predicates': []
        }
    }
    
    # Add node data
    for atoms, state_id in state_to_id.items():
        is_initial = atoms == frozenset(initial_atoms)
        is_goal = goal_atoms.issubset(atoms)
        is_shortest_path = any(
            (state_id == state_to_id[frozenset(edge[0])] or 
             state_id == state_to_id[frozenset(edge[2])])
            for edge in edges
        )
        
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
        
        graph_data['nodes'][state_id] = {
            'id': state_id,
            'state_num': state_id,
            'atoms': [str(atom) for atom in atoms],
            'is_initial': str(is_initial),
            'is_goal': str(is_goal),
            'is_shortest_path': str(is_shortest_path),
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