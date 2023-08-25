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
            'extra_room_table': (8.27387, -6.23233, -0.0678132),
            'extra_room_table_left': (8.27387, -6.211, -0.06700),
            "low_wall_rack": (9.81101, -7.00988, 0.122701),
        }


    def get_obj_attrs(self, obj_name):
        return self.fake_dict[obj_name]


class VLMapWrapper:
    def __init__(self):
        self.vlmap = None

    def get_obj(self, obj_name):
        # use LLM or something to translate the name
        pass