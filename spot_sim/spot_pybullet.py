import pybullet as p
import pybullet_data
import time

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setGravity(0,0,-10)

# Load Spot URDFs
spotStartPos = [0,0,1]
spotStartOrientation = p.getQuaternionFromEuler([0,0,0])
spotId = p.loadURDF("spot_description/spot_base.urdf", spotStartPos, spotStartOrientation)

spotArmStartPos = [0,0,1]
spotArmStartOrientation = p.getQuaternionFromEuler([0,0,0])
spotArmId = p.loadURDF("spot_description/spot_arm.urdf", spotArmStartPos, spotArmStartOrientation)

tableStartPos = [2,0,0]
tableStartOrientation = p.getQuaternionFromEuler([0,0,0])
tableId = p.loadURDF("urdfs/table/table.urdf", tableStartPos, tableStartOrientation)


# Load Other URDFs
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
floorId = p.loadURDF("plane.urdf")


# Start Sim
p.stepSimulation()
spotPos, spotOrn = p.getBasePositionAndOrientation(spotArmId)
print(spotPos, spotOrn)

input()
p.disconnect()