# Branching Visualization Guide for Information Gathering Operators

## Overview

This guide describes the enhanced branching visualization system for mock robot environments that includes information gathering operators. Information gathering operators create decision trees in belief-space planning, where the robot must handle different observation outcomes with appropriate action sequences.

## Key Features

### 🌳 **Branching Path Visualization**
- **Tree Structure**: Shows all information gathering alternatives as a branching tree
- **Decision Points**: Clearly marks where information gathering creates different paths
- **Complete Coverage**: Enumerates all 2^n possible paths for n information gathering points

### 🎯 **Shortest Path Highlighting**
- **Bold Edges**: Optimal path edges are highlighted with thicker, solid lines
- **Visual Distinction**: Non-optimal branches use dashed lines with different colors
- **Interactive Toggle**: Users can show/hide shortest path and alternative branches

### 📐 **Horizontal Layout**  
- **Left-to-Right Flow**: Horizontal tree layout optimized for branching structures
- **Better Readability**: Easier to follow decision trees and sequential operations
- **Tree Navigation**: Natural flow from initial state through decision points to goal

### 📊 **Multi-Format Output**
- **Interactive HTML**: `{task_name}_branching.html` - Interactive tree/graph with controls
- **Structured YAML**: `{task_name}_branching_plan.yaml` - Machine-readable path data
- **Console Output**: Immediate visibility of all complete paths
- **Size Optimization**: Focused graphs are 90%+ smaller than full state spaces

## Information Gathering Detection

The system automatically detects information gathering operators by:

1. **Naming Patterns**: `Inspect*`, `Observe*`, `Check*`, `Scan*`, `Monitor*`
2. **Effect Analysis**: Operators that update belief predicates (`Unknown_X` → `Known_X`)
3. **Precondition Matching**: Multiple operators with same preconditions, different outcomes

## Generated Outputs

### 1. Interactive Branching Visualization

**File**: `{task_name}_branching.html`
**Features**:
- Horizontal tree layout (Left-to-Right)
- Bold shortest path highlighting
- Interactive controls:
  - Toggle shortest path visibility
  - Toggle alternative branches
  - Animate layout changes
  - Reset layout
- Color coding:
  - **Blue bold**: Shortest path edges
  - **Red bold**: Shortest path belief updates  
  - **Light blue dashed**: Alternative branches
  - **Pink dashed**: Alternative belief updates

**Keyboard Shortcuts**:
- `h`: Show help
- `s`: Toggle shortest path
- `e`: Toggle all edges  
- `a`: Toggle animation
- `r`: Reset layout
- `Esc`: Close info panel

### 2. Structured Path Data

**File**: `{task_name}_branching_plan.yaml`
**Contents**:
```yaml
all_complete_paths:
  - path_id: 1
    description: "Optimal Path (All surfaces clean)"
    operators: [list of operators]
    decisions: [key decision points]
    branch_outcomes: [inspection results]
  - path_id: 2
    description: "Hybrid Path (First dirty, second clean)"
    # ... additional paths
```

### 3. Console Output

Immediately prints all complete paths with:
- Path descriptions
- Operator sequences
- Key decision points
- Branch outcomes

## Example: Table Cleaning Task

For a table cleaning task with 2 information gathering points:

### Information Gathering Operators
- `InspectSurfaceClean(robot, table_surface1)` vs `InspectSurfaceDirty(robot, table_surface1)`
- `InspectSurfaceClean(robot, table_surface2)` vs `InspectSurfaceDirty(robot, table_surface2)`

### Generated Paths (2² = 4 paths)

1. **Path 1**: Both surfaces clean (optimal - 10 operators)
2. **Path 2**: Surface 1 dirty, Surface 2 clean (hybrid)  
3. **Path 3**: Surface 1 clean, Surface 2 dirty (hybrid)
4. **Path 4**: Both surfaces dirty (worst case - requires cleaning)

### Visualization Features
- **Horizontal flow**: Initial → Bowl1 removal → Inspect1 → branches → Goal
- **Bold optimal path**: Clear visual hierarchy
- **94% size reduction**: From 718KB to 44KB (focused graph)

## Usage

### Basic Usage
```python
# Standard planning and visualization
creator.plan_and_visualize(initial_atoms, goal_atoms_or, objects, task_name)

# Enhanced branching visualization 
creator.plan_and_visualize_with_branches(initial_atoms, goal_atoms_or, objects, task_name)
```

### Integration with BKLVA
- Compatible with belief-space planning approach
- Works with VLM perception systems
- Supports execution monitoring and replanning
- All branches pre-verified to reach goal

## Technical Implementation

### Core Components

1. **Information Gathering Detection**
```python
def _is_information_gathering_operator(self, operator) -> bool:
    # Detects by naming patterns and belief effects
```

2. **Path Generation**
```python
def _find_branching_paths(self, ...) -> List[Dict]:
    # Complete enumeration of 2^n paths for n branch points
```

3. **Visualization Filtering**
```python
# Only shows nodes and edges part of branching paths
if (source_id, dest_id) in branching_path_edges:
    # Include in focused visualization
```

### Enhanced Edge Styling
```python
edge_data.append({
    'is_shortest_path': is_shortest_path,  # Bold highlighting
    'is_info_gathering': is_info_gathering,  # Special marking
    'affects_belief': affects_belief  # Color coding
})
```

### Horizontal Layout Configuration
```javascript
layout: {
    name: 'breadthfirst',
    rankDir: 'LR',  // Left to Right layout (horizontal)
    nodeSep: 100,
    rankSep: 150,
    roots: '[?is_initial]'  // Start from initial state
}
```

## Benefits

### For Developers
- **Quick Debugging**: See all execution scenarios at once
- **Plan Verification**: Ensure all branches reach goal
- **Performance Analysis**: Compare path lengths and complexity

### For Research
- **Belief-Space Planning**: Visualize information gathering strategies
- **Execution Monitoring**: Understand replanning triggers
- **Algorithm Comparison**: Evaluate different planning approaches

### For Demonstration
- **Clear Presentation**: Horizontal layout shows sequential nature
- **Interactive Exploration**: Stakeholders can toggle different views
- **Complete Coverage**: Shows robot handles all observation outcomes

## Best Practices

### Environment Design
1. **Clear Naming**: Use `Inspect*` for information gathering operators
2. **Belief Predicates**: Structure with `Unknown_*`, `Believe*` patterns
3. **Alternative Outcomes**: Ensure information gathering has meaningful branches

### Visualization Usage
1. **Start with Focused**: Use `_branching.html` for decision tree analysis
2. **Interactive Exploration**: Use controls to focus on relevant paths
3. **Documentation**: Save YAML for quantitative analysis

### Performance Optimization
1. **Focused Graphs**: Only include branching-relevant nodes (90%+ size reduction)
2. **Horizontal Layout**: Better for sequential decision processes
3. **Lazy Loading**: Generate visualizations on demand

## Troubleshooting

### Common Issues

1. **No Branches Detected**
   - Check information gathering operator naming
   - Verify belief predicate effects
   - Ensure alternative operators exist

2. **Layout Problems**
   - Use horizontal layout for tree structures
   - Check node/edge spacing parameters
   - Reset layout if needed

3. **Performance Issues**
   - Use focused branching visualization
   - Limit to relevant paths only
   - Consider file size optimizations

### Debugging Commands

```python
# Check if information gathering detected
print(f"Info gathering operators: {[op for op in operators if self._is_information_gathering_operator(op)]}")

# Verify branching paths
paths = self._find_branching_paths(initial_atoms, goal_atoms_or, objects, transitions)
print(f"Found {len(paths)} branching paths")

# Check file sizes
print(f"Regular: {regular_file_size}, Branching: {branching_file_size}")
```

## Future Enhancements

### Potential Improvements
1. **3D Visualization**: For complex multi-level branching
2. **Probability Weights**: Show likelihood of different branches  
3. **Execution Traces**: Overlay actual execution paths
4. **Comparative Analysis**: Side-by-side algorithm comparison

### Integration Opportunities
1. **VLM Integration**: Show visual perception outputs at decision points
2. **Real Robot**: Overlay actual sensor data and execution
3. **Planning Algorithms**: Compare different belief-space planners
4. **Performance Metrics**: Add timing and success rate analysis

## Conclusion

The enhanced branching visualization system provides comprehensive analysis of information gathering in belief-space planning. With horizontal layout, shortest path highlighting, and focused graph generation, it offers both clarity and performance for understanding robot decision-making under uncertainty.

The system successfully demonstrates how information gathering operators create decision trees, showing the robot's ability to handle different observation outcomes with appropriate action sequences. This is crucial for robust robotics applications where sensor uncertainty requires adaptive planning strategies. 