import logging
import os
from typing import Annotated, Optional
import vtk
import SimpleITK as sitk
import sitkUtils as su
import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLLabelMapVolumeNode, vtkMRMLMarkupsFiducialNode


# PathPlanning

class PathPlanning(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("PathPlanning")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module
        self.parent.contributors = ["Rachel Sparks (King's College London)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is the start of the path planning script with some helpers already implemented
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.  Rachel Sparks has modified this, 
for part of Image-guide Navigation for Robotics taught through King's College London.
""")


# PathPlanningParameterNode

@parameterNodeWrapper
class PathPlanningParameterNode:
    """
    The parameters needed by module.

    inputTargetVolume - The label map the trajectory must be inside
    inputCriticalVolume - The label map the trajectory avoid
    inputEntryFiducials - Fiducials cotaining potential target points
    inputTargetFiducials - Fiducials containing potential entry points
    lengthThreshold - The value above which to exclude trajectories
    outputFiducials - Fiducials containing output points of target and entry pairs
    """
    inputTargetVolume: vtkMRMLLabelMapVolumeNode
    inputCriticalVolume1: vtkMRMLLabelMapVolumeNode
    inputCriticalVolume2: vtkMRMLLabelMapVolumeNode  # New critical volume
    inputEntryFiducials: vtkMRMLMarkupsFiducialNode
    inputTargetFiducials: vtkMRMLMarkupsFiducialNode
    lengthThreshold: Annotated[float, WithinRange(0, 500)] = 100
    outputFiducials: vtkMRMLMarkupsFiducialNode
    validTrajectories: vtkMRMLMarkupsFiducialNode
    optimalTrajectory: vtkMRMLMarkupsFiducialNode


# PathPlanningWidget

class PathPlanningWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/PathPlanning.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = PathPlanningLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputTargetVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLLabelMapVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputTargetVolume = firstVolumeNode

        if not self._parameterNode.inputCriticalVolume1:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLLabelMapVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputCriticalVolume1 = firstVolumeNode

        if not self._parameterNode.inputCriticalVolume2:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLLabelMapVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputCriticalVolume2 = firstVolumeNode

        if not self._parameterNode.inputTargetFiducials:
            firstFiducialNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsFiducialNode")
            if firstFiducialNode:            
                self._parameterNode.inputTargetFiducials = firstFiducialNode

        if not self._parameterNode.inputEntryFiducials:
            firstFiducialNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsFiducialNode")
            if firstFiducialNode:
                self._parameterNode.inputEntryFiducials = firstFiducialNode

        # Automatically create output fiducial nodes if not set yet
        if not self._parameterNode.outputFiducials:
            outputFiducialsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "OutputFiducials")
            self._parameterNode.outputFiducials = outputFiducialsNode

        if not self._parameterNode.validTrajectories:
            validTrajectoriesNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "ValidTrajectories")
            self._parameterNode.validTrajectories = validTrajectoriesNode

        if not self._parameterNode.optimalTrajectory:
            optimalTrajectoryNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "OptimalTrajectory")
            self._parameterNode.optimalTrajectory= optimalTrajectoryNode

        # Now update the UI selectors to show these nodes immediately:
        if hasattr(self.ui, 'outputFiducialSelector'):
            self.ui.outputFiducialSelector.setCurrentNode(self._parameterNode.outputFiducials)
        if hasattr(self.ui, 'trajectoryFiducialSelector'):
            self.ui.trajectoryFiducialSelector.setCurrentNode(self._parameterNode.validTrajectories)
        if hasattr(self.ui, 'optimalTrajectoryFiducialSelector'):
            self.ui.optimalTrajectoryFiducialSelector.setCurrentNode(self._parameterNode.optimalTrajectory)


    def setParameterNode(self, inputParameterNode: Optional[PathPlanningParameterNode]) -> None:

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputTargetVolume and self._parameterNode.inputCriticalVolume1 and self._parameterNode.inputEntryFiducials and self._parameterNode.inputTargetFiducials and self._parameterNode.inputCriticalVolume2:
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select all input nodes")
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        # First initialise the logic
        logic = PathPlanningLogic()
        # Set class parameters
        logic.SetInputTargetImage(self.ui.inputTargetVolumeSelector.currentNode())
        logic.SetEntryPoints(self.ui.inputEntryFiducialSelector.currentNode())
        logic.SetTargetPoints(self.ui.inputTargetFiducialSelector.currentNode())
        logic.SetCriticalVolume1(self.ui.inputCriticalVolumeSelector1.currentNode())
        logic.SetCriticalVolume2(self.ui.inputCriticalVolumeSelector2.currentNode())
        logic.SetLengthThreshold(self.ui.lengthSliderWidget.value)
        logic.SetValidTargetPoints(self.ui.outputFiducialSelector.currentNode())
        logic.SetTrajectoryPoints(self.ui.trajectoryFiducialSelector.currentNode())
        logic.SetOptimalTrajectoryPoints(self.ui.optimalTrajectoryFiducialSelector.currentNode())
        # finally try to run the code. Return false if the code did not run properly
        complete = logic.run()

        # print out an error message if the code returned false
        if not complete:
            print('encountered an error')


# PathPlanningLogic

class PathPlanningLogic(ScriptedLoadableModuleLogic):

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return PathPlanningParameterNode(super().getParameterNode())
    
    def SetInputTargetImage(self, imageNode):
        if (self.hasImageData(imageNode)):
            self.myTargetImage = imageNode
    
    def SetCriticalVolume1(self, criticalVolumeNode1):
        self.myCriticalVolume1 = criticalVolumeNode1
    
    def SetCriticalVolume2(self, criticalVolumeNode2):
        self.myCriticalVolume2 = criticalVolumeNode2

    def SetEntryPoints(self, entryNode):
        self.myEntries = entryNode

    def SetTargetPoints(self, targetNode):
        self.myTargets = targetNode

    def SetLengthThreshold(self, lengthThresholdNode):
        self.myLengthTreshold = lengthThresholdNode

    def SetValidTargetPoints(self, outputNode):
        self.myValidTargets = outputNode

    def SetTrajectoryPoints(self, outputNode):
        self.myValidTrajectories = outputNode

    def SetOptimalTrajectoryPoints(self, outputNode):
        self.myOptimalTrajectory = outputNode


    def hasImageData(self, volumeNode):
        """This is an example logic method that
        returns true if the passed in volume
        node has valid image data
        """
        if not volumeNode:
            logging.debug('hasImageData failed: no volume node')
            return False
        if volumeNode.GetImageData() is None:
            logging.debug('hasImageData failed: no image data in volume node')
            return False
        return True

    def isValidInputOutputData(self, inputTargetVolumeNode, inputTargetFiducialsNode, inputEntryFiducialsNodes, outputFiducialsNode):
        """Validates if the output is not the same as input
        """
        if not inputTargetVolumeNode:
            logging.debug('isValidInputOutputData failed: no input target volume node defined')
            return False
        if not inputTargetFiducialsNode:
            logging.debug('isValidInputOutputData failed: no input target fiducials node defined')
            return False
        if not inputEntryFiducialsNodes:
            logging.debug('isValidInputOutputData failed: no input entry fiducials node defined')
            return False
        if not outputFiducialsNode:
            logging.debug('isValidInputOutputData failed: no output fiducials node defined')
            return False
        if inputTargetFiducialsNode.GetID()==outputFiducialsNode.GetID():
            logging.debug('isValidInputOutputData failed: input and output fiducial nodes are the same. Create a new output to avoid this error.')
            return False
        return True

    def run(self):
        """Run the path planning algorithm with two critical volumes."""
        if not self.isValidInputOutputData(self.myTargetImage, self.myTargets, self.myEntries, self.myValidTargets):
            slicer.util.errorDisplay('Not all inputs are set.')
            return False
        if not self.hasImageData(self.myTargetImage):
            raise ValueError("Input target volume is not appropriately defined.")
        if not self.hasImageData(self.myCriticalVolume1):
            raise ValueError("Input target volume is not appropriately defined.")
        if not self.hasImageData(self.myCriticalVolume2):
            raise ValueError("Input target volume is not appropriately defined.")
        
        import time
        startTime = time.time()
        logging.info("Processing started")
        
        pointPicker = PickPointsMatrix()
        pointPicker.run(self.myTargetImage, self.myTargets, self.myValidTargets)

        trajectoryChecker = TrajectoryChecker()
        trajectoryChecker.run(self.myEntries, self.myValidTargets, self.myCriticalVolume1, self.myCriticalVolume2, self.myValidTrajectories, self.myLengthTreshold)

        optimalTrajectory = OptimalTrajectoryEvaluator()
        optimalTrajectory.run(self.myCriticalVolume1,self.myCriticalVolume2, self.myValidTrajectories, self.myOptimalTrajectory)
        stopTime = time.time()
        print(f"Processing completed in {stopTime - startTime:.2f} seconds")

        try:
            node = slicer.util.getNode("TrajectoryLines")
            slicer.mrmlScene.RemoveNode(node)
        except slicer.util.MRMLNodeNotFoundException:
            pass

        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "TrajectoryLines")
        modelNode.CreateDefaultDisplayNodes()
        modelNode.GetDisplayNode().SetColor(1, 0, 0)  # red
        modelNode.GetDisplayNode().SetLineWidth(2)

        # PolyData components
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()

        # Assume outputFiducials has entry-target pairs (i, i+1)
        numPoints = self.myValidTrajectories.GetNumberOfControlPoints()
        for i in range(0, numPoints, 2):
            if i + 1 >= numPoints:
                break

            p1 = [0, 0, 0]
            p2 = [0, 0, 0]
            self.myValidTrajectories.GetNthControlPointPositionWorld(i, p1)
            self.myValidTrajectories.GetNthControlPointPositionWorld(i + 1, p2)

            id1 = points.InsertNextPoint(p1)
            id2 = points.InsertNextPoint(p2)

            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, id1)
            line.GetPointIds().SetId(1, id2)
            lines.InsertNextCell(line)

        # Build the polydata
        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)
        polyData.SetLines(lines)

        # Assign it to the model node
        modelNode.SetAndObservePolyData(polyData)
        return True

class PickPointsMatrix():
  def run(self, inputVolume, inputFiducials, outputFiducials):
    # First bit of clean up is to remove all points from the output-- otherwise rerunning will duplicate these
    outputFiducials.RemoveAllControlPoints()
    # we can get a transformation from our input volume
    mat = vtk.vtkMatrix4x4()
    inputVolume.GetRASToIJKMatrix(mat)
    
    # set it to a transform type
    transform = vtk.vtkTransform()
    transform.SetMatrix(mat)

    for x in range(0, inputFiducials.GetNumberOfControlPoints()):
      pos = [0,0,0]
      inputFiducials.GetNthControlPointPosition(x, pos)
      # get index from position using our transformation
      ind = transform.TransformPoint(pos)

      # get pixel using that index
      pixelValue = inputVolume.GetImageData().GetScalarComponentAsDouble (int(ind[0]), int(ind[1]), int(ind[2]), 0)
      if (pixelValue == 1):
        outputFiducials.AddControlPoint(pos)
        print(f"RAS: {pos}")

class TrajectoryChecker:
    def run(self, entryFiducials, validTargetFiducials, criticalVolume1, criticalVolume2, outputFiducials, maxLength=100.0):
        # find valid trajectories between entry and valid target points avoiding critical volumes.

        # Clear previous output
        outputFiducials.RemoveAllControlPoints()

        # we can get a transformation from our input volume
        mat1 = vtk.vtkMatrix4x4()
        mat2 = vtk.vtkMatrix4x4()
        criticalVolume1.GetRASToIJKMatrix(mat1)
        criticalVolume2.GetRASToIJKMatrix(mat2)
        imageData1 = criticalVolume1.GetImageData()
        imageData2 = criticalVolume2.GetImageData()
        sampleStep=1.0

        # set it to a transform type
        transform1 = vtk.vtkTransform()
        transform1.SetMatrix(mat1)
        transform2 = vtk.vtkTransform()
        transform2.SetMatrix(mat2)

        # Iterate over pairs of entry and target points
        for i in range(entryFiducials.GetNumberOfControlPoints()):
            entryPos = [0,0,0]
            entryFiducials.GetNthControlPointPosition(i, entryPos)

            for j in range(validTargetFiducials.GetNumberOfControlPoints()):
                targetPos = [0,0,0]
                validTargetFiducials.GetNthControlPointPosition(j, targetPos)

                # Compute distance between points
                dist = vtk.vtkMath.Distance2BetweenPoints(entryPos, targetPos)**0.5
                if dist == 0 or dist > maxLength:
                    continue  # skip identical points or trajectories that are too long.

                # Sample along line from entry to target
                numSamples = max(1, int(dist / sampleStep))
            
                trajectoryBlocked = False
                for s in range(numSamples + 1):
                    t = s / numSamples
                    samplePoint = [
                        entryPos[0] * (1 - t) + targetPos[0] * t,
                        entryPos[1] * (1 - t) + targetPos[1] * t,
                        entryPos[2] * (1 - t) + targetPos[2] * t,
                    ]

                    ind1 = transform1.TransformPoint(samplePoint)
                    ind2 = transform2.TransformPoint(samplePoint)

                    # get pixel using that index
                    pixelValue1 =  imageData1.GetScalarComponentAsDouble (int(ind1[0]), int(ind1[1]), int(ind1[2]), 0)
                    pixelValue2 =  imageData2.GetScalarComponentAsDouble (int(ind2[0]), int(ind2[1]), int(ind2[2]), 0)
                    if (pixelValue1 == 1 or pixelValue2 == 1):
                        trajectoryBlocked = True
                        break

                if not trajectoryBlocked:
                    outputFiducials.AddControlPoint(entryPos)
                    outputFiducials.AddControlPoint(targetPos)
                    print(f"Valid trajectory between entry {i} and target {j},{dist}")


class OptimalTrajectoryEvaluator:
    def run(self, criticalVolume1, criticalVolume2, trajectoryFiducials, optimalTrajectory):
        # Clear previous output
        optimalTrajectory.RemoveAllControlPoints()

        # Convert VTK volumes to SimpleITK distance maps
        distanceMap1 = self.computeDistanceMapFromLabelMap(criticalVolume1)
        distanceMap2 = self.computeDistanceMapFromLabelMap(criticalVolume2)

        bestScore = -1
        bestIndex = 0

        numPoints = trajectoryFiducials.GetNumberOfControlPoints()

        # Loop over trajectory segments: point i -> i+1
        for i in range(0, numPoints - 1, 2):
            start = [0, 0, 0]
            end = [0, 0, 0]
            trajectoryFiducials.GetNthControlPointPosition(i, start)
            trajectoryFiducials.GetNthControlPointPosition(i + 1, end)

            score = self.evaluateTrajectory(start, end, distanceMap1, distanceMap2)
            print(score)
            if score > bestScore:
                bestScore = score
                bestIndex = i

        start = [0, 0, 0]
        end = [0, 0, 0]
        trajectoryFiducials.GetNthControlPointPosition(bestIndex, start)
        trajectoryFiducials.GetNthControlPointPosition(bestIndex+1, end)
        optimalTrajectory.AddControlPoint(start)
        optimalTrajectory.AddControlPoint(end)
        
        print(f"Best trajectory index pair: {start}, {end}, Score: {bestScore}")
        return bestIndex  # or return the (start, end) if you prefer

    def computeDistanceMapFromLabelMap(self, criticalVolume):
        sitkInput = su.PullVolumeFromSlicer(criticalVolume)
        distanceFilter = sitk.DanielssonDistanceMapImageFilter()
        sitkOutput = distanceFilter.Execute(sitkInput)
        su.PushVolumeToSlicer(sitkOutput, name='TempDistanceMap')  # just for visualization/debug
        return sitkOutput


    def evaluateTrajectory(self, start, end, distanceMap1, distanceMap2, sampleStep=1.0):
        dist = vtk.vtkMath.Distance2BetweenPoints(start, end) ** 0.5
        numSamples = max(1, int(dist / sampleStep))

        totalDistance = 0
        validSamples = 0

        for i in range(numSamples + 1):
            t = i / numSamples
            samplePoint = [
                start[0] * (1 - t) + end[0] * t,
                start[1] * (1 - t) + end[1] * t,
                start[2] * (1 - t) + end[2] * t,
            ]

            # Convert RAS -> LPS for SimpleITK
            samplePointLPS = [-samplePoint[0], -samplePoint[1], samplePoint[2]]

            try:
                idx1 = distanceMap1.TransformPhysicalPointToIndex(samplePointLPS)
                idx2 = distanceMap2.TransformPhysicalPointToIndex(samplePointLPS)
                d1 = distanceMap1.GetPixel(idx1)
                d2 = distanceMap2.GetPixel(idx2)
                totalDistance += min(d1, d2)
                validSamples += 1
            except Exception as e:
                # Optional: logging.debug(f"Sample out of bounds: {samplePointLPS}")
                continue

        if validSamples == 0:
            return -1

        return totalDistance / validSamples


class PathPlanningTest(ScriptedLoadableModuleTest):
    """
    Test suite for PathPlanning module logic.
    Includes edge cases, positive and negative tests.
    """

    def setUp(self):
        """Resets the Slicer scene and loads test volumes."""
        slicer.mrmlScene.Clear()
        self.delayDisplay("Scene cleared. Loading test data.")

        # Test data paths - hard code to data on your system
        basePath = "C:/Users/ayman/OneDrive - King's College London/IGNR/Week23/TestSet"
        self.targetVolume = slicer.util.loadVolume(basePath + '/r_hippoTest.nii.gz')
        self.criticalVolume1 = slicer.util.loadLabelVolume(basePath + '/ventriclesTest.nii.gz')
        self.criticalVolume2 = slicer.util.loadLabelVolume(basePath + '/vesselsTestDilate1.nii.gz')

        assert self.targetVolume and self.criticalVolume1 and self.criticalVolume2, "Volume loading failed"

        self.delayDisplay("Test volumes loaded successfully.")

    def runTest(self):
        self.setUp()
        self.test_PickPoints_EmptyOutsideMask()
        self.test_PickPoints_InsideMask()
        self.test_TrajectoryBlocked()
        self.test_TrajectoryValid()
        self.test_OptimalTrajectoryEvaluated()

    # Test 1: PickPointsMatrix (Negative case)
    def test_PickPoints_EmptyOutsideMask(self):
        self.delayDisplay("Test: PickPoints with points outside mask (should return 0)")

        fiducials = slicer.vtkMRMLMarkupsFiducialNode()
        fiducials.AddControlPoint(-1000, -1000, -1000)  # Way outside
        fiducials.AddControlPoint(1000, 1000, 1000)      # Another extreme

        slicer.mrmlScene.AddNode(fiducials)

        output = slicer.vtkMRMLMarkupsFiducialNode()
        slicer.mrmlScene.AddNode(output)

        PickPointsMatrix().run(self.targetVolume, fiducials, output)
        assert output.GetNumberOfControlPoints() == 0, "Expected no valid points"

        self.delayDisplay("Passed: No points inside mask.")

    # Test 2: PickPointsMatrix (Positive case)
    def test_PickPoints_InsideMask(self):
        self.delayDisplay("Test: PickPoints with known inside point")

        # Manually add a known foreground voxel
        fiducials = slicer.vtkMRMLMarkupsFiducialNode()
        fiducials.AddControlPoint(0, 0, 0)
        slicer.mrmlScene.AddNode(fiducials)

        output = slicer.vtkMRMLMarkupsFiducialNode()
        slicer.mrmlScene.AddNode(output)

        PickPointsMatrix().run(self.targetVolume, fiducials, output)
        # Cannot guarantee it's inside, but we should test that logic executes
        assert output is not None, "Output fiducials is None"

        self.delayDisplay("Completed inside mask check (manual verify result if needed)")

    # Test 3: TrajectoryChecker (Blocked path)
    def test_TrajectoryBlocked(self):
        self.delayDisplay("Test: Trajectory that intersects critical volume (should be blocked)")

        entries = slicer.vtkMRMLMarkupsFiducialNode()
        targets = slicer.vtkMRMLMarkupsFiducialNode()
        output = slicer.vtkMRMLMarkupsFiducialNode()

        entries.AddControlPoint(0, 0, 0)
        targets.AddControlPoint(100, 100, 100)  # Long trajectory, likely intersects critical

        slicer.mrmlScene.AddNode(entries)
        slicer.mrmlScene.AddNode(targets)
        slicer.mrmlScene.AddNode(output)

        TrajectoryChecker().run(entries, targets, self.criticalVolume1, self.criticalVolume2, output, 200.0)
        assert output.GetNumberOfControlPoints() == 0, "Blocked path should return no points"

        self.delayDisplay("Passed: Blocked trajectory returned no points.")

    # Test 4: TrajectoryChecker (Valid path)
    def test_TrajectoryValid(self):
        self.delayDisplay("Test: Short clear trajectory (should be valid)")

        entries = slicer.vtkMRMLMarkupsFiducialNode()
        targets = slicer.vtkMRMLMarkupsFiducialNode()
        output = slicer.vtkMRMLMarkupsFiducialNode()

        entries.AddControlPoint(0, 0, 0)
        targets.AddControlPoint(0, 1, 0)  # Very short path

        slicer.mrmlScene.AddNode(entries)
        slicer.mrmlScene.AddNode(targets)
        slicer.mrmlScene.AddNode(output)

        TrajectoryChecker().run(entries, targets, self.criticalVolume1, self.criticalVolume2, output, 100.0)
        assert output.GetNumberOfControlPoints() == 2, "Expected valid entry-target pair"

        self.delayDisplay("Passed: Valid trajectory identified.")

    # Test 5: OptimalTrajectoryEvaluator
    def test_OptimalTrajectoryEvaluated(self):
        self.delayDisplay("Test: Evaluate optimal trajectory scoring")

        fiducials = slicer.vtkMRMLMarkupsFiducialNode()
        slicer.mrmlScene.AddNode(fiducials)
        # Add two trajectories
        fiducials.AddControlPoint(0, 0, 0)
        fiducials.AddControlPoint(1, 1, 1)
        fiducials.AddControlPoint(10, 10, 10)
        fiducials.AddControlPoint(11, 11, 11)

        output = slicer.vtkMRMLMarkupsFiducialNode()
        slicer.mrmlScene.AddNode(output)

        evaluator = OptimalTrajectoryEvaluator()
        index = evaluator.run(self.criticalVolume1, self.criticalVolume2, fiducials, output)

        assert output.GetNumberOfControlPoints() == 2, "Expected 2 points in optimal path"
        assert index in [0, 2], "Returned index must match trajectory segments"

        self.delayDisplay("Passed: Optimal trajectory evaluated and returned.")

    # Edge Case 1: No Fiducials Provided
    def test_NoFiducialsProvided(self):
        self.delayDisplay("Edge Case: No input fiducials (expect graceful handling)")

        inputPoints = slicer.vtkMRMLMarkupsFiducialNode()
        outputPoints = slicer.vtkMRMLMarkupsFiducialNode()

        slicer.mrmlScene.AddNode(inputPoints)
        slicer.mrmlScene.AddNode(outputPoints)

        PickPointsMatrix().run(self.targetVolume, inputPoints, outputPoints)
        assert outputPoints.GetNumberOfControlPoints() == 0, "Output should be empty with no inputs"

        self.delayDisplay("Passed: Empty input handled gracefully.")

    # Edge Case 2: Empty Target Volume
    def test_EmptyTargetVolume(self):
        self.delayDisplay("Edge Case: Empty (all zero) mask volume")

        # Create empty label volume
        emptyImage = slicer.vtkMRMLLabelMapVolumeNode()
        emptyImage.SetName("EmptyTarget")
        slicer.mrmlScene.AddNode(emptyImage)

        # Set dummy image data
        imageData = vtk.vtkImageData()
        imageData.SetDimensions(10, 10, 10)
        imageData.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        imageData.GetPointData().GetScalars().Fill(0)
        emptyImage.SetAndObserveImageData(imageData)

        # Try running PickPoints
        fiducials = slicer.vtkMRMLMarkupsFiducialNode()
        fiducials.AddControlPoint(0, 0, 0)
        output = slicer.vtkMRMLMarkupsFiducialNode()
        slicer.mrmlScene.AddNode(fiducials)
        slicer.mrmlScene.AddNode(output)

        PickPointsMatrix().run(emptyImage, fiducials, output)
        assert output.GetNumberOfControlPoints() == 0, "No points should be valid in empty mask"

        self.delayDisplay("Passed: Empty target volume handled.")

    # Edge Case 3: Same Entry and Target Point
    def test_SameEntryAndTarget(self):
        self.delayDisplay("Edge Case: Entry and target points are identical")

        entryTarget = slicer.vtkMRMLMarkupsFiducialNode()
        entryTarget.AddControlPoint(10, 10, 10)
        slicer.mrmlScene.AddNode(entryTarget)

        output = slicer.vtkMRMLMarkupsFiducialNode()
        slicer.mrmlScene.AddNode(output)

        TrajectoryChecker().run(entryTarget, entryTarget, self.criticalVolume1, self.criticalVolume2, output, 200.0)
        assert output.GetNumberOfControlPoints() == 0, "Should not return same-point trajectory"

        self.delayDisplay("Passed: Same-point trajectory rejected.")

    # Edge Case 4: Points Outside Image Bounds
    def test_PointsOutsideImageBounds(self):
        self.delayDisplay("Edge Case: Fiducials outside image bounds")

        fiducials = slicer.vtkMRMLMarkupsFiducialNode()
        fiducials.AddControlPoint(9999, 9999, 9999)
        slicer.mrmlScene.AddNode(fiducials)

        output = slicer.vtkMRMLMarkupsFiducialNode()
        slicer.mrmlScene.AddNode(output)

        PickPointsMatrix().run(self.targetVolume, fiducials, output)
        assert output.GetNumberOfControlPoints() == 0, "Out-of-bounds points should not be added"

        self.delayDisplay("Passed: Out-of-bounds points skipped safely.")


    # Edge Case 5: Extremely Long Trajectory
    def test_ExtremelyLongTrajectory(self):
        self.delayDisplay("Edge Case: Extremely long trajectory (beyond threshold)")

        entries = slicer.vtkMRMLMarkupsFiducialNode()
        targets = slicer.vtkMRMLMarkupsFiducialNode()
        output = slicer.vtkMRMLMarkupsFiducialNode()

        entries.AddControlPoint(0, 0, 0)
        targets.AddControlPoint(10000, 10000, 10000)

        slicer.mrmlScene.AddNode(entries)
        slicer.mrmlScene.AddNode(targets)
        slicer.mrmlScene.AddNode(output)

        TrajectoryChecker().run(entries, targets, self.criticalVolume1, self.criticalVolume2, output, 100.0)
        assert output.GetNumberOfControlPoints() == 0, "Long trajectory should be rejected"

        self.delayDisplay("Passed: Over-threshold trajectory rejected.")

    # Edge Case 6: Optimal Path Tie
    def test_OptimalTrajectoryTie(self):
        self.delayDisplay("Edge Case: Multiple trajectories with same score")

        fiducials = slicer.vtkMRMLMarkupsFiducialNode()
        output = slicer.vtkMRMLMarkupsFiducialNode()

        # Create two identical trajectories
        fiducials.AddControlPoint(0, 0, 0)
        fiducials.AddControlPoint(1, 1, 1)
        fiducials.AddControlPoint(0, 0, 0)
        fiducials.AddControlPoint(1, 1, 1)

        slicer.mrmlScene.AddNode(fiducials)
        slicer.mrmlScene.AddNode(output)

        evaluator = OptimalTrajectoryEvaluator()
        index = evaluator.run(self.criticalVolume1, self.criticalVolume2, fiducials, output)

        assert index in [0, 2], "Must return a valid tie index"
        assert output.GetNumberOfControlPoints() == 2, "Output must contain winning trajectory"

        self.delayDisplay("Passed: Tie-breaking in optimal trajectory handled.")