# Spot RGB-D Data Collection Example

## Overview

This example demonstrates collecting synchronized RGB-D data from Spot's hand camera with corresponding pose information. Key features include:

- Manual or automatic image capture
- RGB and depth image synchronization
- Camera pose recording
- Movement detection to prevent blur
- Visualization generation
- Organized data storage

## Prerequisites

1. **Hardware**
   - Spot robot with arm
   - Hand camera
   - Computer with Python 3.11+

2. **Software**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup**
   ```bash
   export BOSDYN_CLIENT_USERNAME="your_username"
   export BOSDYN_CLIENT_PASSWORD="your_password"
   ```

## Usage

### Basic Command
```bash
python src/robots/spot/examples/spot_collect_rgbd.py --hostname <robot_ip>
```

### Arguments
```bash
--hostname        # Robot IP address (required)
--manual_images   # "True" for manual, "False" for automatic (default: "True")
--path_dir        # Base save directory (default: "../data")
--dir_name        # Data subdirectory name (default: "Spot")
```

### Data Structure
```
data/
└── Spot/
    ├── RGB/           # RGB images (.npy)
    │   └── rgb_img_*.npy
    ├── Depth/         # Depth images (.npy)
    │   └── depth_img_*.npy
    ├── Pose/          # Camera poses (.npy)
    │   └── pose_*.npy
    └── Visualized/    # Visualizations (.jpg)
        ├── RGB_*.jpg
        └── RGB-D_*.jpg
```

## Data Formats

### RGB Images
- Format: NumPy array (.npy)
- Shape: (H, W, 3)
- Type: uint8
- Color space: RGB

### Depth Images
- Format: NumPy array (.npy)
- Shape: (H, W)
- Type: float32
- Units: Meters
- Range: Typically 0.0 to 10.0

### Pose Data
- Format: NumPy array (.npy)
- Shape: (7,)
- Content: [x, y, z, qw, qx, qy, qz]
- Frame: Vision frame
- Units: Meters for position, normalized quaternion for rotation

## Features

### Movement Detection
- Tracks robot movement between frames
- Skips captures when movement detected
- Configurable movement threshold

### Visualization
- RGB-D overlay generation
- Depth colorization using jet colormap
- JPEG format for easy viewing

### Error Handling
- Connection verification
- Image capture validation
- Graceful interruption handling
- Informative error messages

## Example Code

### Manual Collection
```python
# Initialize configuration
config = DataCollectionConfig(
    manual_images=True,
    path_dir="../data",
    dir_name="Spot"
)

# Connect to robot
with robot_connection(hostname) as robot:
    # Collect data
    image_client = robot.ensure_client(ImageClient.default_service_name)
    # ... collection loop
```

### Automatic Collection
```python
config = DataCollectionConfig(
    manual_images=False,
    pic_hz=2.0,  # 2 Hz capture rate
    movement_threshold=0.2  # meters
)
```

## Best Practices

1. **Data Collection**
   - Ensure good lighting
   - Keep camera clean
   - Maintain stable position
   - Monitor storage space

2. **Error Handling**
   - Check connection status
   - Validate image quality
   - Monitor robot state
   - Handle interruptions gracefully

3. **Resource Management**
   - Release robot lease
   - Close connections properly
   - Clean up temporary files 