import os
import pybullet as p
import numpy as np

class CustomRobot:
    def __init__(self, urdf_path):
        p.setAdditionalSearchPath(os.path.dirname(urdf_path))
        self.robot_id = p.loadURDF(os.path.basename(urdf_path), useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)

        self.joints = [i for i in range(p.getNumJoints(self.robot_id)) if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED]
        self.dof = len(self.joints)
        self.reset_state()

    def reset_state(self):
        for j in self.joints:
            p.resetJointState(self.robot_id, j, targetValue=0)
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id, jointIndices=self.joints,
                                     controlMode=p.VELOCITY_CONTROL, forces=[0. for _ in self.joints])

    def get_dof(self):
        return self.dof

    def get_position_and_velocity(self):
        joint_states = p.getJointStates(self.robot_id, self.joints)
        joint_pos = [state[0] for state in joint_states]
        joint_vel = [state[1] for state in joint_states]
        return joint_pos, joint_vel

    def calculate_inverse_kinematics(self, position, orientation):
        return p.calculateInverseKinematics(self.robot_id, self.joints[-1], position, orientation)[:self.dof]

    def calculate_inverse_dynamics(self, pos, vel, desired_acc):
        assert len(pos) == len(vel) and len(vel) == len(desired_acc)
        return list(p.calculateInverseDynamics(self.robot_id, pos, vel, desired_acc))

    def set_target_positions(self, desired_pos):
        assert len(desired_pos) == len(self.joints)
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id, jointIndices=self.joints,
                                     controlMode=p.POSITION_CONTROL, targetPositions=desired_pos)

    def set_torques(self, desired_torque):
        assert len(desired_torque) == len(self.joints)
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id, jointIndices=self.joints,
                                     controlMode=p.TORQUE_CONTROL, forces=desired_torque)

    def set_group_target_positions(self, group_pos):
        assert len(group_pos) == 3
        # Define the joint groups
        group_x = [0, 3, 6, 9]
        group_y = [1, 4, 7, 10]
        group_d = [2, 5, 8]

        # Create a list for all joint positions
        desired_pos = [0] * len(self.joints)

        # Assign the group positions
        for idx in group_x:
            desired_pos[idx] = group_pos[0]
        for idx in group_y:
            desired_pos[idx] = group_pos[1]
        for idx in group_d:
            desired_pos[idx] = group_pos[2]

        # Set the target positions for all joints
        self.set_target_positions(desired_pos)

    def apply_joint_forces(self, joint_forces):
        assert len(joint_forces) == len(self.joints)
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id, jointIndices=self.joints,
                                     controlMode=p.TORQUE_CONTROL, forces=joint_forces)

    def calculate_tcp_position(self):
        # Get the state of the last link, which should be the fixed joint from which the offset is applied
        
        end_effector_state = p.getLinkState(self.robot_id, self.get_dof() - 1)

        
        position = np.array(end_effector_state[4])            # Global position of the link
        orientation = end_effector_state[5]                   # Global orientation (quaternion) of the link

        rotation_matrix = p.getMatrixFromQuaternion(orientation)

        # Get the Z-direction vector (normal vector) in the global frame
        z_direction = np.array([-rotation_matrix[6], -rotation_matrix[7], rotation_matrix[8]])

        # Specify the offset along the Z-axis
        offset = 0.08 # Offset from last joint to the Tip position
        #offset = 0.015

        # Calculate the TCP position by applying the offset
        tcp_position = position + offset * z_direction
        return tcp_position
    
    def get_contact_points(self):
        contact_points = p.getContactPoints(bodyA=self.robot_id)
        contact_forces = []

        for point in contact_points:
            normalforce = point[9]  # normal force
            lateralforce1 = point[10]  # first lateral friction force
            lateralforce2 = point[11]  # second lateral friction force
            contact_forces.append((normalforce, lateralforce1, lateralforce2))

        return contact_forces
