import Sofa
import Sofa.Core
from Sofa.constants import *
import os
import atexit
import numpy as np
import random
import csv
import sys

pathToDatas = os.path.dirname(os.path.abspath(__file__)) + '/gui/control/datas/'

def actuator_readRequestNewPressure(pathToDatas, actuator_number):
    filename = f"actuator{actuator_number}_requestNewPressure.txt"
    filepath = os.path.join(pathToDatas, filename)
    try:
        with open(filepath, 'r') as file:
            data = file.read()
        return data
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None

class ActuatorDataWriter:
    def __init__(self, pathToDatas, actuator_number):
        self.pathToDatas = pathToDatas
        self.actuator_number = actuator_number

    def write_data(self, data_type, data):
        filename = f'{self.pathToDatas}actuator{self.actuator_number}_{data_type}Values.txt'
        try:
            with open(filename, 'w') as file:
                data_str = str(data)  # Convert the number to a string
                file.write(data_str + '\n')
            return 0
        except Exception as e:
            print(f"Failed to write to {filename}: {e}")
            return -1

class ActuatorController(Sofa.Core.Controller):
    def __init__(self, *a, **kw):
        Sofa.Core.Controller.__init__(self, *a, **kw)
        self.node = kw["node"]
        self.isComplete = False
        self.constraints = []
        self.dofs = []
        self.totalTime = 0

        self.holdTime = 0.3
        self.elapsedTimeSinceLastChange = 0.0

        self.pressureRef = self.generate_pressure_profiles()
        self.num_episodes = len(self.pressureRef)
        self.steps_per_episode = len(self.pressureRef[0]) if self.pressureRef.size > 0 else 0

        self.current_episode = 0
        self.current_step = 0

        self.dt = 0.01

        # Open a CSV file to write the data
        self.dataFilePath = os.path.dirname(os.path.abspath(__file__)) + "/DynamicsSweep.csv"
        self.dataFile = open(self.dataFilePath, "w", newline='')
        self.csvWriter = csv.writer(self.dataFile)
        self.csvWriter.writerow(["Time", "Pressure1", "Pressure2", "Pressure3", "ForceX", "ForceY", "ForceZ", "AvgPosX", "AvgPosY", "AvgPosZ", "TrackedPosX1", "TrackedPosY1", "TrackedPosZ1", "TrackedPosX2", "TrackedPosY2", "TrackedPosZ2", "TrackedPosX3", "TrackedPosY3", "TrackedPosZ3", "AvgVelX", "AvgVelY", "AvgVelZ"])

        atexit.register(self.cleanup)

        # Position Velocity Initialization
        self.trackedIndices = [465, 442, 453]  # 3 tracked indices
        self.previousPositions = None

        TripodNode = self.node.Modelling.getChild('Tripod')
        Bellows_Node = TripodNode.getChild("ElasticBody")
        self.dofs = Bellows_Node.MechanicalModel.dofs

        for i in range(3):
            cavityNode = Bellows_Node.MechanicalModel.getChild('cavity' + str(i + 1))
            self.constraints.append(cavityNode.SurfacePressureConstraint)

        # Create a ConstantForceField dynamically
        self.force_field = Bellows_Node.MechanicalModel.addObject('ConstantForceField', indices=self.trackedIndices, forces=[[0,0,0], [0,0,0], [0,0,0]], showArrowSize='0.001')

    def generate_pressure_profiles(self):
        num_profiles = 1000
        duration = 10.0  # seconds
        timestep = 0.5   # seconds
        time_steps = int(duration / timestep)
        pressure_profiles = np.zeros((num_profiles, time_steps, 3))
        min_pressure = -20000
        max_pressure = 35000

        for profile in range(num_profiles):
            for i in range(3):
                pressure_profiles[profile, :, i] = np.random.uniform(min_pressure, max_pressure, time_steps)
        return pressure_profiles

    def onAnimateBeginEvent(self, e):
        self.totalTime += self.dt
        self.elapsedTimeSinceLastChange += self.dt

        pressures = [0, 0, 0]
        force_value_x = 0.0
        force_value_y = 0.0
        force_value_z = 0.0
        force_values = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

        if self.elapsedTimeSinceLastChange >= self.holdTime:
            self.elapsedTimeSinceLastChange = 0.0

            if self.current_step < self.steps_per_episode:
                pressures = self.pressureRef[self.current_episode][self.current_step]
                for i, constraint in enumerate(self.constraints):
                    constraint.value = [pressures[i] / 1000]

                force_value_x = np.random.choice(np.linspace(0, 10000, 100))
                force_value_y = np.random.choice(np.linspace(0, 10000, 100))
                force_value_z = np.random.choice(np.linspace(0, 10000, 100))
                force_values = [
                    [force_value_x ,force_value_y, force_value_z],
                    [force_value_x ,force_value_y, force_value_z],
                    [force_value_x ,force_value_y, force_value_z]
                ]
                self.force_field.forces.value = force_values

                self.current_step += 1
            else:
                self.current_step = 0
                self.current_episode += 1
                if self.current_episode >= self.num_episodes:
                    self.current_episode = 0

        # Save data at every time step
        dt = self.node.findData('dt').value
        TripodNode = self.node.Modelling.getChild('Tripod')
        Bellows_Node = TripodNode.getChild("ElasticBody")

        mechanicalObject_modelNode = Bellows_Node.MechanicalModel.dofs

        positions = mechanicalObject_modelNode.position.array()
        velocities = mechanicalObject_modelNode.velocity.array()

        trackedPositions = positions[self.trackedIndices]
        trackedVelocities = velocities[self.trackedIndices]

        avgPosition = np.mean(trackedPositions, axis=0)
        avgVelocity = np.mean(trackedVelocities, axis=0)
        fltrackedPositions = trackedPositions.flatten()

        Time_rounded = round(self.totalTime, 3)
        self.csvWriter.writerow([Time_rounded] + list(pressures) + [force_value_x, force_value_y, force_value_z] + avgPosition.tolist() + fltrackedPositions.tolist() + avgVelocity.tolist())

        print(f"{Time_rounded}, {pressures}, {force_value_x}, {force_value_y}, {force_value_z}, {avgPosition.tolist()}, {avgVelocity.tolist()}\n")

    def cleanup(self):
        if not self.dataFile.closed:
            self.dataFile.close()
            print("File closed and cleanup done.")
