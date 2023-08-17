import pybullet as p
import time
import pybullet_data
import ray


@ray.remote
class Simulator(object):
    def __init__(self):
        self.physicsClient = p.connect(p.DIRECT)  # Use DIRECT mode for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeId = p.loadURDF("plane.urdf")
        self.cubeStartPos = [0, 0, 1]
        self.cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.cubeId = p.loadURDF("cube_small.urdf", self.cubeStartPos, self.cubeStartOrientation)

    def step(self):
        p.stepSimulation()
        time.sleep(1. / 240.)

    def get_cube_position_and_orientation(self):
        return p.getBasePositionAndOrientation(self.cubeId)


@ray.remote
def run_simulation(simulator, steps):
    for _ in range(steps):
        simulator.step.remote()


if __name__ == "__main__":
    ray.init()
    simulator = Simulator.remote()

    run_simulation.remote(simulator, 10000)

    # Do other stuff...

    cube_position_and_orientation = ray.get(simulator.get_cube_position_and_orientation.remote())
    print(cube_position_and_orientation)

    ray.shutdown()
