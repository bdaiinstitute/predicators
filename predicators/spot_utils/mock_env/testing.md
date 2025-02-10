# Testing Guide for Mock Spot Environment

This guide covers testing practices and guidelines for the mock Spot environment.

## Test Organization

### Environment Tests (`test_mock_spot_env.py`)

1. Basic Properties:
   ```python
   def test_mock_spot_env():
       # Test environment initialization
       env = MockSpotEnv(data_dir=temp_dir)
       assert env.get_name() == "mock_spot"
       
       # Test types
       assert len(env.types) == 5
       robot_type = next(t for t in env.types if t.name == "robot")
       
       # Test predicates
       assert len(env.predicates) == 25
       
       # Test operators
       assert len(env.strips_operators) == 5
   ```

2. State Management:
   ```python
   def test_state_management():
       # Test state creation
       state_id = env.add_state(rgbd=None, gripper_open=True)
       assert state_id in env._observations
       
       # Test state updates
       env.update_state(state_id, gripper_open=False)
       assert not env._observations[state_id].gripper_open
   ```

3. Transition Graph:
   ```python
   def test_transitions():
       # Test valid transitions
       env.add_transition(state_1, "PickObjectFromTop", state_2)
       
       # Test invalid transitions
       with pytest.raises(ValueError):
           env.add_transition("invalid", "PickObjectFromTop", state_2)
   ```

### Perceiver Tests (`test_mock_spot_perceiver.py`)

1. Image Handling:
   ```python
   def test_image_handling():
       # Test saving images
       perceiver.save_image(mock_rgbd)
       
       # Test loading images
       loaded_rgbd = perceiver.get_observation().rgbd
       assert np.array_equal(loaded_rgbd.rgb, mock_rgbd.rgb)
   ```

2. State Tracking:
   ```python
   def test_state_tracking():
       # Test state updates
       perceiver.update_state(gripper_open=True,
                            objects_in_view={"cup"})
       obs = perceiver.get_observation()
       assert obs.gripper_open
       assert "cup" in obs.objects_in_view
   ```

### Task Tests (`test_mock_spot_task.py`)

1. Task Creation:
   ```python
   def test_task_creation():
       # Test task initialization
       task = MockSpotTask()
       task.set_environment(env)
       
       # Test goal generation
       goal = task.generate_goal_description()
       assert isinstance(goal, GoalDescription)
   ```

2. State Conversion:
   ```python
   def test_state_conversion():
       # Test observation to state conversion
       obs = perceiver.get_observation()
       state = task.observation_to_state(obs)
       assert isinstance(state, State)
   ```

## Test Fixtures

1. Environment Setup:
   ```python
   @pytest.fixture
   def mock_env():
       # Create temporary directory
       temp_dir = tempfile.mkdtemp()
       try:
           # Initialize environment
           env = MockSpotEnv(data_dir=temp_dir)
           yield env
       finally:
           # Clean up
           shutil.rmtree(temp_dir)
   ```

2. Mock Data:
   ```python
   @pytest.fixture
   def mock_rgbd():
       return RGBDImageWithContext(
           rgb=np.zeros((100, 100, 3), dtype=np.uint8),
           depth=np.zeros((100, 100), dtype=np.float32),
           camera_info=None
       )
   ```

## Test Categories

### Unit Tests

1. Component Tests:
   - Test individual methods
   - Test error handling
   - Test edge cases

2. State Tests:
   - Test state creation/deletion
   - Test state updates
   - Test state validation

3. Transition Tests:
   - Test valid transitions
   - Test invalid transitions
   - Test transition validation

### Integration Tests

1. Environment-Perceiver Integration:
   ```python
   def test_env_perceiver_integration():
       # Test environment with perceiver
       env = MockSpotEnv(data_dir=temp_dir)
       perceiver = MockSpotPerceiver(data_dir=temp_dir)
       
       # Add state with image
       state_id = env.add_state(rgbd=mock_rgbd)
       
       # Verify perceiver can access image
       obs = perceiver.get_observation()
       assert obs.rgbd is not None
   ```

2. Task-Environment Integration:
   ```python
   def test_task_env_integration():
       # Test task with environment
       env = MockSpotEnv(data_dir=temp_dir)
       task = MockSpotTask()
       task.set_environment(env)
       
       # Test task execution
       obs = env.reset("train", 0)
       action = task.get_action(obs)
       next_obs, reward, done = env.step(action)
   ```

### System Tests

1. End-to-End Tests:
   ```python
   def test_end_to_end():
       # Test complete workflow
       env = MockSpotEnv(data_dir=temp_dir)
       task = MockSpotTask()
       perceiver = MockSpotPerceiver(data_dir=temp_dir)
       
       # Create states and transitions
       states = create_test_states(env)
       create_test_transitions(env, states)
       
       # Execute task
       execute_test_task(env, task)
   ```

2. Performance Tests:
   ```python
   def test_performance():
       # Test with large state space
       env = MockSpotEnv(data_dir=temp_dir)
       for i in range(1000):
           env.add_state(...)
           
       # Test memory usage
       import psutil
       process = psutil.Process()
       mem_before = process.memory_info().rss
       
       # Perform operations
       # ...
       
       # Check memory usage
       mem_after = process.memory_info().rss
       assert (mem_after - mem_before) < threshold
   ```

## Best Practices

1. Test Setup:
   - Use temporary directories
   - Clean up after tests
   - Use fixtures for common setup

2. Test Organization:
   - Group related tests
   - Use descriptive test names
   - Follow test hierarchy

3. Test Coverage:
   - Test success cases
   - Test failure cases
   - Test edge cases
   - Test performance

4. Test Maintenance:
   - Keep tests up to date
   - Remove obsolete tests
   - Document test requirements

## Running Tests

1. Run all tests:
   ```bash
   pytest tests/
   ```

2. Run specific test file:
   ```bash
   pytest tests/test_mock_spot_env.py
   ```

3. Run with coverage:
   ```bash
   pytest --cov=predicators tests/
   ```

4. Run with verbose output:
   ```bash
   pytest -v tests/
   ``` 