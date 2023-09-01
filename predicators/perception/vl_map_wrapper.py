import logging
from typing import Dict, Optional, Set, Tuple

import numpy as np

from predicators import utils


class FakeVLMapWrapper:
    def __init__(self):
        # TODO note
        # we can put some fake objects here for test
        # some useful properties: hierarchy, spatial reference,

        # Predefine some object names here
        self.fake_dict = {
            # TODO put default objects here for reference
            # "tool_room_table": (6.63041, -6.35143, 0.179613),
            # "extra_room_table": (8.27387, -6.23233, -0.0678132),
            # "low_wall_rack": (9.81101, -7.00988, 0.122701),
            # "high_wall_rack": (9.81101, -7.00988, 0.757881268568327966),
            # "bucket":
            # (7.043112552148553, -8.198686802340527, -0.18750694527153725),
            # "platform": (8.63513, -7.87694, -0.0751688),

            # TODO hardcode some positions here
            'extra_room_table_left': (8.63513, -7.87694, -0.0751688),

            'vlmap_extra_room_table_left': (9.81101, -7.00988, 0.122701),
        }


    def get_obj_attrs(self, obj_name):
        return self.fake_dict[obj_name]


class VLMapWrapper:
    def __init__(self):
        self.vlmap = None

    def get_obj(self, obj_name):
        # use LLM or something to translate the name
        pass