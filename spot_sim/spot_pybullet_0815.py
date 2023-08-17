import pickle
from collections import defaultdict

import pybullet as p
import pybullet_data
import time

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setGravity(0, 0, -10)

with open('./test_state_dict.pickle', 'rb') as fp:
    test_state_dict = pickle.load(fp)

spot_name2urdfs = {
    "spot:robot": ["spot_description/spot_base.urdf", "spot_description/spot_arm.urdf"],
    # "spot:robot": ["spot_description/spot_gripper.urdf"],  # TODO replace with Spot gripper only
    "tool_room_table:flat_surface": ["urdfs/table/table.urdf"],
    "extra_room_table:flat_surface": ["urdfs/table/table.urdf"],
    "low_wall_rack:flat_surface": [],  # TODO find one or skip for now
    "high_wall_rack:flat_surface": [],  # TODO find one or skip for now
    "bucket:bag": [],  # TODO find a random one
    "platform:platform": [],  # TODO find a random one
    "floor:floor": ["plane.urdf"],
    "hammer:tool": [],  # TODO find a hammer
    # TODO to find more tools
}

p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF

# Start to populate the simulator
name2id = defaultdict(list)

for obj, value in test_state_dict.items():
    print(obj, value['x'], value['y'], value['z'])

    obj_str = str(obj)

    urdf_list = spot_name2urdfs[obj_str]  # TODO it's a Struct Object type!
    for urdf in urdf_list:
        pos = [value['x'], value['y'], value['z']]
        orientation = p.getQuaternionFromEuler([0., 0., 0.])  # TODO no orientation available

        obj_id = p.loadURDF(urdf, pos, orientation)
        name2id[obj_str].append(obj_id)

# Start Sim
p.stepSimulation()
# spotPos, spotOrn = p.getBasePositionAndOrientation(spotArmId)
spotPos, spotOrn = p.getBasePositionAndOrientation(name2id["spot:robot"][1])
print(spotPos, spotOrn)

input()
p.disconnect()
