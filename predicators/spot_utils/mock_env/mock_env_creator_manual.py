"""Manual mock environment creator for testing.

This class provides functionality to:
- Create mock environments from manually collected images
- Process RGB and depth images
- Create states from multiple image views
"""

from typing import Dict, Optional, Tuple, Literal, Set

import numpy as np
from predicators.structs import Object
from predicators.spot_utils.mock_env.mock_env_creator_base import MockEnvCreatorBase
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

    pass