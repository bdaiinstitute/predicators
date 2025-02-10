# Implementation Note: Adding New Belief Space Operators

This guide outlines the process of adding new belief space operators and predicates to the mock Spot environment.

## Example: Container Location Emptiness

We'll use the example of adding predicates and operators for determining if a container/location is empty.

### 1. Add New Predicates

In `mock_spot_env.py`, add these predicates:

```python
# Belief predicates for container emptiness
_Unknown_ContainerEmpty = Predicate("Unknown_ContainerEmpty", [_container_type], _dummy_classifier)
_Known_ContainerEmpty = Predicate("Known_ContainerEmpty", [_container_type], _dummy_classifier)
_BelieveTrue_ContainerEmpty = Predicate("BelieveTrue_ContainerEmpty", [_container_type], _dummy_classifier)
_BelieveFalse_ContainerEmpty = Predicate("BelieveFalse_ContainerEmpty", [_container_type], _dummy_classifier)
```

Add them to `BELIEF_PREDICATES` set:

```python
BELIEF_PREDICATES.update({
    _Unknown_ContainerEmpty,
    _Known_ContainerEmpty,
    _BelieveTrue_ContainerEmpty,
    _BelieveFalse_ContainerEmpty
})
```

### 2. Add New Operators

In `mock_spot_env.py`'s `_create_operators()`, add these operators:

```python
# ObserveContainerContent: Observe if a container is empty
parameters = [
    Variable("?robot", _robot_type),
    Variable("?container", _container_type),
]

preconditions = {
    LiftedAtom(_Unknown_ContainerEmpty, [parameters[1]]),
}

add_effects = {
    LiftedAtom(_Known_ContainerEmpty, [parameters[1]]),
}

delete_effects = {
    LiftedAtom(_Unknown_ContainerEmpty, [parameters[1]]),
}

yield STRIPSOperator("ObserveContainerContent",
                     parameters,
                     preconditions,
                     add_effects,
                     delete_effects,
                     set())

# OpenDrawerFindEmpty: Open drawer and find it empty
parameters = [
    Variable("?robot", _robot_type),
    Variable("?container", _container_type),
]

preconditions = {
    LiftedAtom(_Unknown_ContainerEmpty, [parameters[1]]),
}

add_effects = {
    LiftedAtom(_Known_ContainerEmpty, [parameters[1]]),
    LiftedAtom(_BelieveTrue_ContainerEmpty, [parameters[1]]),
}

delete_effects = {
    LiftedAtom(_Unknown_ContainerEmpty, [parameters[1]]),
}

yield STRIPSOperator("OpenDrawerFindEmpty",
                     parameters,
                     preconditions,
                     add_effects,
                     delete_effects,
                     set())

# OpenDrawerFindNotEmpty: Open drawer and find objects
parameters = [
    Variable("?robot", _robot_type),
    Variable("?container", _container_type),
]

preconditions = {
    LiftedAtom(_Unknown_ContainerEmpty, [parameters[1]]),
}

add_effects = {
    LiftedAtom(_Known_ContainerEmpty, [parameters[1]]),
    LiftedAtom(_BelieveFalse_ContainerEmpty, [parameters[1]]),
}

delete_effects = {
    LiftedAtom(_Unknown_ContainerEmpty, [parameters[1]]),
}

yield STRIPSOperator("OpenDrawerFindNotEmpty",
                     parameters,
                     preconditions,
                     add_effects,
                     delete_effects,
                     set())
```

### 3. Update Tests

Create test cases in `test_mock_env_drawer_compare.py`:

1. Test observing empty container:
   - Initial state: Container emptiness unknown
   - Action: ObserveContainerContent
   - Final state: Container emptiness known

2. Test opening drawer and finding it empty:
   - Initial state: Container emptiness unknown
   - Action: OpenDrawerFindEmpty
   - Final state: Container emptiness known, believed empty

3. Test opening drawer and finding objects:
   - Initial state: Container emptiness unknown
   - Action: OpenDrawerFindNotEmpty
   - Final state: Container emptiness known, believed not empty

### 4. Naming Conventions

Follow these naming patterns for belief space predicates:

- `Unknown_X`: Initial state of not knowing X
- `Known_X`: State after observing X
- `BelieveTrue_X`: Belief that X is true after observation
- `BelieveFalse_X`: Belief that X is false after observation

### 5. General Process for Adding New Belief Space Features

1. Identify the belief property (e.g., container emptiness)
2. Add predicates following naming convention:
   - `Unknown_X`
   - `Known_X`
   - `BelieveTrue_X`
   - `BelieveFalse_X`
3. Add to `BELIEF_PREDICATES` set
4. Create operators:
   - Observation operator
   - Action operators that affect beliefs
5. Add test cases:
   - Initial unknown state
   - Observation action
   - Final known state with belief
6. Update documentation

### 6. Testing Process[text](implementation_plan.md)

1. Create test file if needed
2. Add test cases for each operator
3. Verify transition graphs:
   - Initial state has Unknown predicate
   - Actions correctly transition states
   - Final state has Known and Believe predicates
4. Test combinations with other operators 