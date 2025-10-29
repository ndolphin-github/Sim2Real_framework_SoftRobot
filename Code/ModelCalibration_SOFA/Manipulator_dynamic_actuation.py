import os 
import math
import Sofa
import Sofa.Gui
import Sofa.Core
import numpy as np
import Sofa.Simulation
import atexit
from PluginList import pluginList

from stlib3.scene import Scene
from stlib3.physics.mixedmaterial import Rigidify
from stlib3.components import addOrientedBoxRoi
from splib3.numerics import vec3
from splib3.numerics.quat import Quat

from DynamicController import ActuatorController

meshpath = os.path.dirname(os.path.abspath(__file__))
VTKMesh = os.path.join(meshpath, 'mesh', 'Assembly', 'ManipulatorWhole25.vtk')
STLMesh = os.path.join(meshpath, 'mesh', 'Assembly', 'ManipulatorWhole25.stl')
CavityMesh = os.path.join(meshpath, 'mesh', 'Assembly', 'cavity_sim.STL')

def ElasticBody(name="ElasticBody", Volume_mesh=VTKMesh, Surface_mesh=STLMesh, Cavity_Mesh=CavityMesh):
    self = Sofa.Core.Node(name)
    mechanicalmodel = self.addChild("MechanicalModel")
    mechanicalmodel.addObject('MeshVTKLoader', name='loader', filename=Volume_mesh, rotation=[0,0,0], translation=[0,0,0], scale=1 )
    mechanicalmodel.addObject('MeshTopology', src='@loader', name='container')
    
    mechanicalmodel.addObject('MechanicalObject', name='dofs', template='Vec3', showIndices=False, showIndicesScale=0.015)
    
    mechanicalmodel.addObject('UniformMass', totalMass=1.10)
    mechanicalmodel.addObject('TetrahedronFEMForceField', name='linearElasticBehavior', poissonRatio=0.41, youngModulus=3000) # 4500, 6000
  
    mechanicalmodel.addObject('CGLinearSolver', iterations=25, tolerance=1e-9, threshold=1e-9)
    mechanicalmodel.addObject('GenericConstraintSolver', maxIterations=500, tolerance=1e-8)
    bottomBoxROI = [75, 75, 5, -75, -75, -1]
    mechanicalmodel.addObject('BoxROI', name='boxROI', box=bottomBoxROI, doUpdate=False, drawBoxes=False)
    mechanicalmodel.addObject('RestShapeSpringsForceField', points='@boxROI.indices', stiffness=1e12, angularStiffness=1e12)
    
    mechanicalmodel.addObject('TriangleCollisionModel', simulated='1', contactStiffness='100', selfCollision='1', group='1')
    mechanicalmodel.addObject('LineCollisionModel', selfCollision=True)
    mechanicalmodel.addObject('PointCollisionModel', selfCollision=True)
    
    visualmodel = Sofa.Core.Node("VisualModel")
    visualmodel.addObject('MeshSTLLoader', name='loader', filename=Surface_mesh, rotation=[0,0,0], translation=[0,0,0], scale=1)
    visualmodel.addObject('OglModel', src=visualmodel.loader.getLinkPath(), name='renderer', color=[0.9, 0.8, 0.8, 0.8])
    self.addChild(visualmodel)
    visualmodel.addObject('BarycentricMapping', input=mechanicalmodel.dofs.getLinkPath(), output=visualmodel.renderer.getLinkPath())
    
    for i in range(3):
        cavity_rotation_angles = math.pi / 180 * (-90 - 120 * i)
        z_offset_cavity = 0
        R = 50
        translation_cavity = [R * math.cos(cavity_rotation_angles), R * math.sin(cavity_rotation_angles), z_offset_cavity]
        cavity = mechanicalmodel.addChild('cavity' + str(i + 1))
        cavity.addObject('MeshSTLLoader', name='cavityLoader', filename=CavityMesh)  
        cavity.addObject('MeshTopology', src='@cavityLoader', name='cavityTopo')
        cavity.addObject('MechanicalObject', name='cavity', translation=translation_cavity, rotation=[0, 0, 0], scale=1)
        cavity.addObject('SurfacePressureConstraint', name='SurfacePressureConstraint', template='Vec3', value=0.001, triangles='@topo.triangles', valueType='pressure')
        cavity.addObject('BarycentricMapping', name='mapping', mapForces=False, mapMasses=False)
        
        collisionModel = mechanicalmodel.addChild('collisionCavity' + str(i + 1))
        collisionModel.addObject('MeshSTLLoader', name='cavityLoader', filename=CavityMesh, translation=translation_cavity, rotation=[0, 0, 0])
        collisionModel.addObject('MeshTopology', src='@cavityLoader', name='topo')
        collisionModel.addObject('MechanicalObject', name='collisMech')
        collisionModel.addObject('TriangleCollisionModel', selfCollision=True)
        collisionModel.addObject('LineCollisionModel', selfCollision=False)
        collisionModel.addObject('PointCollisionModel', selfCollision=False)
        collisionModel.addObject('BarycentricMapping', name='collision mapping')


    
    return self

def Tripod(name='Tripod'):
    self = Sofa.Core.Node(name)
    self.addChild(ElasticBody())
    
    def rigidifyFrame(self):
        deformableObject = self.ElasticBody.MechanicalModel
        self.ElasticBody.init()
        translation_CP = [0, 0, 174.5]
        
        box = addOrientedBoxRoi(self, position=[list(i) for i in deformableObject.dofs.rest_position.value], name="FixingBoxROI", translation=translation_CP, eulerRotation=[0, 0, 0], scale=[170, 170, 6])
        box.init()
        box.drawBoxes = False

        groupIndices = []
        frames = []
        groupIndices.append([ind for ind in box.indices.value])
        frames.append(translation_CP + list(Quat.createFromEuler([0, 0, 0], inDegree=True)))
        
        for i in range(6):
            translation_RB = [0, -6, 25.25 + 25 * i]
            boxRib = addOrientedBoxRoi(self, position=[list(i) for i in deformableObject.dofs.rest_position.value], name="RibFixingBoxROI", translation=translation_RB, eulerRotation=[0, 0, 0], scale=[65, 60, 5])
            boxRib.init()
            boxRib.drawBoxes = False
            groupIndices.append([ind for ind in boxRib.indices.value])
            frames.append(translation_RB + list(Quat.createFromEuler([0, 0, 0], inDegree=True)))
             
        rigidifiedstruct = Rigidify(self, deformableObject, groupIndices=groupIndices, frames=frames, name="RigidifiedStructure")
    
    rigidifyFrame(self)    
    return self

def createScene(rootNode):
    scene = Scene(rootNode, gravity=[+9810.0, 0.0, 0.0], iterative=False, plugins=pluginList)
    scene.addMainHeader()
    scene.addObject('DefaultVisualManagerLoop')
    scene.addObject('FreeMotionAnimationLoop', name='animationLoop')
    scene.addObject('GenericConstraintSolver', maxIterations=50, tolerance=1e-5)
    scene.Simulation.addObject('GenericConstraintCorrection')
    scene.Settings.mouseButton.stiffness = 10
    scene.Simulation.TimeIntegrationSchema.rayleighStiffness = 0.05
    scene.VisualStyle.displayFlags = "showBehavior"
    scene.dt = 0.01
    scene.addObject('BackgroundSetting', color=[0, 0, 0, 1])
    
    tripod = scene.Modelling.addChild(Tripod()) 
    scene.Simulation.addChild(tripod)
    
    
    scene.addObject(ActuatorController(node=rootNode))
    return scene

def main():
    root = Sofa.Core.Node("root")
    createScene(root)
    Sofa.Simulation.init(root)
    Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 800)
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()

if __name__ == '__main__':
    main()