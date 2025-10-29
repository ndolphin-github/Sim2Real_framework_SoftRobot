import pybullet as p
import pybullet_data
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from CustomModel import CustomRobot
import os

def objective_function(group_pos, robot, target_position):
  
    robot.set_group_target_positions(group_pos)
    
    # Simulate more steps to allow better convergence
    for _ in range(200):  
        p.stepSimulation()

    # Calculate the final position
    current_position = robot.calculate_tcp_position()
    
    # Return the error (distance) between current TCP position and target
    error = np.linalg.norm(current_position - target_position)
    return error

def get_joint_limits():
    return [(-0.2, 0.2),  # thetaX
            (-0.2, 0.2),  # thetaY
            (0.0, 0.02)]  # thetad 

def optimize_joint_angles(robot, target_position, max_iterations=500, threshold=1e-9):
    """
    Perform global optimization first, followed by local optimization to refine the solution.
    """
    joint_limits = get_joint_limits()
    bounds = [(low, high) for low, high in joint_limits]

    # Global optimization using differential evolution
    global_opt_result = differential_evolution(objective_function, bounds, args=(robot, target_position), 
                                               maxiter=300, tol=threshold)
    
    # Local optimization using the result of global optimization as an initial guess
    result = minimize(objective_function, global_opt_result.x, args=(robot, target_position), 
                      method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': max_iterations, 'ftol': threshold, 'disp': True})
    
    return result.x

def process_positions(positions):
    """
    Process each target position by optimizing joint angles to match the robot's TCP position.
    """
    results = []
    urdf_path = "RobotModel/SurrogateModel.urdf"

    # Connect to PyBullet and load the robot
    physics_client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    robot = CustomRobot(urdf_path)

    for position in positions:
        target_position = np.array(position)
        
        # Optimize the joint angles to match the target position
        optimized_angles = optimize_joint_angles(robot, target_position)
        
        # Calculate the predicted position after optimization
        predicted_position = robot.calculate_tcp_position()
        
        # Calculate the error between predicted and target positions
        error = np.linalg.norm(predicted_position - target_position)
        
        # Save the result
        results.append(list(target_position) + list(optimized_angles) + list(predicted_position) + [error])
    
    p.disconnect()
    return results

def main():
    """
    Main function to process the real data positions and run the optimization.
    """
    real_data = pd.read_csv("Pressure_vs_Pos.csv")  # Predefined target positions
    positions = real_data[['predicted_position1', 'predicted_position2', 'predicted_position3']].values

    # Process the positions and get the optimized joint angles
    results = process_positions(positions)

    # Save the results to a CSV file
    columns = ['x', 'y', 'z'] + ['theta_x', 'theta_y', 'd'] + ['pred_x', 'pred_y', 'pred_z', 'error']
    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_csv('Real_Estimated_Joint_Positions.csv', index=False)
    print("Results saved")

if __name__ == "__main__":
    main()
