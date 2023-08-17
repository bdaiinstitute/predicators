import pickle
from collections import defaultdict

import pybullet as p
import pybullet_data
import time

from urdf_models import models_data

from multiprocessing import Process, Queue

# Start Sim
# p.stepSimulation()
# # spotPos, spotOrn = p.getBasePositionAndOrientation(spotArmId)
# spotPos, spotOrn = p.getBasePositionAndOrientation(name2id["spot:robot"][1])
# print(spotPos, spotOrn)


def simulator(queue):

    # cubeStartPos = [0, 0, 1]
    # cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    # cubeId = p.loadURDF("cube_small.urdf", cubeStartPos, cubeStartOrientation)

    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    p.setGravity(0, 0, -10)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF

    with open('./test_state_dict.pickle', 'rb') as fp:
        test_state_dict = pickle.load(fp)

    # TODO a small lib for PyBullet models
    pb_models = models_data.model_lib()
    pb_namelist = pb_models.model_name_list

    spot_name2urdfs = {
    "spot:robot": ["spot_description/spot_base.urdf", "spot_description/spot_arm.urdf"],
    # "spot:robot": ["spot_description/spot_arm.urdf"],
    # "spot:robot": ["spot_description/spot_gripper.urdf"],  # TODO replace with Spot gripper only
        # "tool_room_table:flat_surface": ["urdfs/table/table.urdf"],
        "tool_room_table:flat_surface": ["urdfs/table/table2.urdf"],
        "extra_room_table:flat_surface": ["urdfs/table/table2.urdf"],
        "low_wall_rack:flat_surface": [],  # TODO find one or skip for now
        "high_wall_rack:flat_surface": [],  # TODO find one or skip for now
        "bucket:bag": [pb_models['cracker_box']],  # TODO find a random one
        "platform:platform": [],  # TODO find a random one
        "floor:floor": ["plane.urdf"],
        "hammer:tool": [],  # TODO find a hammer
        # TODO to find more tools
    }

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

    while True:
        if not queue.empty():
            command = queue.get()
            # Process the command
            print(f"Received command: {command}")

            if command == 'q':
                break

            elif command == 'print':
                spotPos, spotOrn = p.getBasePositionAndOrientation(name2id["spot:robot"][1])
                print(spotPos, spotOrn)

            elif command == 'add':
                pos = [2, 2, 0]
                orientation = p.getQuaternionFromEuler([0., 0., 0.])  # TODO no orientation available
                # urdf = "urdfs/table/table.urdf"
                urdf = "cube_small.urdf"

                obj_id = p.loadURDF(urdf, pos, orientation)
                print("added test object", obj_id)

            else:
                print(f'Command not implemented: {command}')

        p.stepSimulation()
        time.sleep(1. / 240.)

    p.disconnect()


def monitor(queue):
    while True:
        # Monitor for external commands and put them in the queue
        command = input("Enter a command: ")
        queue.put(command)

        time.sleep(0.1)


if __name__ == "__main__":
    queue = Queue()
    simulator_process = Process(target=simulator, args=(queue,))

    simulator_process.start()

    monitor(queue)

    simulator_process.join()

# input()
# p.disconnect()
