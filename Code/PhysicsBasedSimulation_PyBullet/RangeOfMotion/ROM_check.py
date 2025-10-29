import pybullet as p
import pybullet_data
import os
import numpy as np
import pandas as pd
from CustomModel import CustomRobot 

def print_link_origins(robot_id, num_links):
    for i in range(num_links):
        link_state = p.getLinkState(robot_id, i)
        print(f"Link {i}:")
        print(f"  World Position: {link_state[4]}")
        print(f"  World Orientation: {link_state[5]}")

def save_joint_state(file_path, joint_states):
    df = pd.DataFrame(joint_states, columns=['Position_X', 'Position_Y', 'Position_Z', 'Normal_X', 'Normal_Y', 'Normal_Z'])
    df.to_csv(file_path, index=False)

# Connect to PyBullet
physics_client = p.connect(p.GUI)  # Use GUI mode to visualize the robot's movement

# Disable real-time simulation
p.setRealTimeSimulation(0)

# Set search path and gravity
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)  # Gravity should be negative in the z-axis

# Set up the camera
cam_distance, cam_yaw, cam_pitch, cam_xyz_target = 0.2, 0, 0, [0.0, 0.0, 0.1]
p.resetDebugVisualizerCamera(
    cameraDistance=cam_distance,
    cameraYaw=cam_yaw,
    cameraPitch=cam_pitch,
    cameraTargetPosition=cam_xyz_target
)

# Load URDF
rootpath = os.getcwd()
#urdf_path = os.path.join(rootpath, "RobotModel\SurrogateModel_withToolTip.urdf")
urdf_path = os.path.join(rootpath, "RobotModel\SurrogateModel.urdf")
robot = CustomRobot(urdf_path)

# Read the CSV file
csv_file = 'joint_angles.csv'
data = pd.read_csv(csv_file)

joint_states = []

# Iterate through each row in the CSV
for index, row in data.iterrows():
    # Extract group positions from CSV
    group_pos = [row['thetaX'], row['thetaY'], row['d']] 
    robot.set_group_target_positions(group_pos)
    
    # Step simulation to update the joint states
    for _ in range(100):  # Adjust the number of steps to ensure stability
        p.stepSimulation()
        #time.sleep(1/1000)  # Add a small sleep to slow down the simulation for visualization

    # Get link state of the last link (index 10)
    last_joint_state = p.getLinkState(robot.robot_id, 10)
    orientation_quat = last_joint_state[5]  # World Link Frame Orientation

    # Calculate rotation matrix from quaternion
    rotation_matrix = p.getMatrixFromQuaternion(orientation_quat)

    # Extract normal (Z-axis) vector from the rotation matrix
    normal_vector_global = np.array([rotation_matrix[6], rotation_matrix[7], rotation_matrix[8]])

    tcp_position = robot.calculate_tcp_position() 
    
    # Concatenate position and normalized normal vector into a single array
    tcp_pose = np.concatenate((tcp_position, normal_vector_global))
    joint_states.append(tcp_pose)
 
    if index % 100 == 0:  # Print progress every 100 lines
        print(f"Processed {index} lines")

# Save the joint states to a new CSV file
output_file = 'ROM_100K.csv'
save_joint_state(output_file, joint_states)

# Disconnect from PyBullet
p.disconnect()
