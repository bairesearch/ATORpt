"""ATORpt_RFpropertiesClass.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt RF Properties - RF Properties transformations (primitive space: ellipse or tri/artificial ellipse)

"""

import torch as pt
import copy

from ATORpt_RFglobalDefs import *
import ATORpt_RFellipsePropertiesClass
import ATORpt_RFoperations

class RFpropertiesClass(ATORpt_RFellipsePropertiesClass.EllipsePropertiesClass):
	def __init__(self, resolutionIndex, resolutionFactor, imageSize, RFtype, centerCoordinates, axesLength, angle, colour):
		self.resolutionIndex = resolutionIndex
		self.resolutionFactor = resolutionFactor
		self.imageSize = imageSize	#stored as width,height (note torchvision, openCV, TFA, etc typically use a height,width standard so imageSize parameters are often swapped when passed to functions)
		self.RFtype = RFtype
		super().__init__(centerCoordinates, axesLength, angle, colour)
		if RFtype == RFtypeTri:
			self.vertexCoordinatesRelative = deriveTriVertexCoordinatesFromArtificialEllipseProperties(axesLength, angle)
		self.isColourFilter = True
		self.numberOfDimensions = 2
		self.filterIndex = None
		self.imageSegmentIndex = None
		self.numberOfKernels = None	#numberRFfiltersInTensor

def deriveTriVertexCoordinatesFromArtificialEllipseProperties(axesLength, angle):
	angle1 = angle
	angle2 = angle + 90
	angle3 = angle - 90
	vertexCoordinatesRelative1 = ATORpt_RFoperations.calculateRelativePositionGivenAngleAndLength(angle1, axesLength[0])
	vertexCoordinatesRelative2 = ATORpt_RFoperations.calculateRelativePositionGivenAngleAndLength(angle2, axesLength[1])
	vertexCoordinatesRelative3 = ATORpt_RFoperations.calculateRelativePositionGivenAngleAndLength(angle3, axesLength[1])
	vertexCoordinatesRelative = [vertexCoordinatesRelative1, vertexCoordinatesRelative2, vertexCoordinatesRelative3]
	return vertexCoordinatesRelative

def printRFproperties(RFproperties):
	print("printRFproperties: numberOfDimensions = ", RFproperties.numberOfDimensions, ", resolutionIndex = ", RFproperties.resolutionIndex, ", isColourFilter = ", RFproperties.isColourFilter, ", imageSize = ", RFproperties.imageSize)
	if RFproperties.RFtype == RFtypeEllipse:
		ATORpt_RFellipsePropertiesClass.printEllipseProperties(RFproperties)
	elif RFproperties.RFtype == RFtypeTri:
		ATORpt_RFellipsePropertiesClass.printEllipseProperties(RFproperties)
		print("vertexCoordinatesRelative = ", RFproperties.vertexCoordinatesRelative)


def generateRFtransformedProperties(neuronComponent, RFpropertiesParent):
	if RFpropertiesParent.numberOfDimensions == 2:
		return generateRFtransformedProperties2D(neuronComponent, RFpropertiesParent)
	elif RFpropertiesParent.numberOfDimensions == 3:
		return generateRFtransformedProperties3D(neuronComponent, RFpropertiesParent)

def generateRFtransformedProperties2D(neuronComponent, RFpropertiesParent):
	RFtransformedProperties = copy.copy(neuronComponent.RFproperties)
	RFtransformedProperties.centerCoordinates = transformPoint2D(neuronComponent.RFproperties.centerCoordinates, RFpropertiesParent)
	endCoordinates = calculateEndCoordinatesPosition2D(neuronComponent)
	endCoordinates = transformPoint2D(endCoordinates, RFpropertiesParent)
	RFtransformedProperties.axesLength = ATORpt_RFoperations.calculateDistance2D(RFtransformedProperties.centerCoordinates, endCoordinates)
	RFtransformedProperties.angle = neuronComponent.RFproperties.angle - RFpropertiesParent.angle
	return RFtransformedProperties

def generateRFtransformedProperties3D(neuronComponent, RFpropertiesParent):
	RFtransformedProperties = copy.copy(neuronComponent.RFproperties)
	RFtransformedProperties.centerCoordinates = transformPoint3D(neuronComponent.RFproperties.centerCoordinates, RFpropertiesParent)
	endCoordinates = calculateEndCoordinatesPosition3D(neuronComponent)
	endCoordinates = transformPoint3D(endCoordinates, RFpropertiesParent)
	RFtransformedProperties.axesLength = ATORpt_RFoperations.calculateDistance3D(RFtransformedProperties.centerCoordinates, endCoordinates)
	RFtransformedProperties.angle = ((neuronComponent.RFproperties.angle[0] - RFpropertiesParent.angle[0]), (neuronComponent.RFproperties.angle[1] - RFpropertiesParent.angle[1]))
	return RFtransformedProperties

def transformPoint2D(coordinates, RFpropertiesParent):
	coordinatesTransformed = (coordinates[0] - RFpropertiesParent.centerCoordinates[0], coordinates[1] - RFpropertiesParent.centerCoordinates[1])
	coordinatesRelativeAfterRotation = ATORpt_RFoperations.calculateRelativePosition2D(RFpropertiesParent.angle, RFpropertiesParent.axesLength[0])
	coordinatesTransformed = (coordinatesTransformed[0] - coordinatesRelativeAfterRotation[0], coordinatesTransformed[1] - coordinatesRelativeAfterRotation[1])
	coordinatesTransformed = (coordinates[0] / RFpropertiesParent.axesLength[0], coordinates[1] / RFpropertiesParent.axesLength[1])
	return coordinatesTransformed

def transformPoint3D(coordinates, RFpropertiesParent):
	coordinatesTransformed = (coordinates[0] - RFpropertiesParent.centerCoordinates[0], coordinates[1] - RFpropertiesParent.centerCoordinates[1], coordinates[2] - RFpropertiesParent.centerCoordinates[2])
	coordinatesRelativeAfterRotation = ATORpt_RFoperations.calculateRelativePosition3D(RFpropertiesParent.angle, RFpropertiesParent.axesLength[0])
	coordinatesTransformed = (coordinatesTransformed[0] - coordinatesRelativeAfterRotation[0], coordinatesTransformed[1] - coordinatesRelativeAfterRotation[1], coordinatesTransformed[2] - coordinatesRelativeAfterRotation[2])
	coordinatesTransformed = (coordinates[0] / RFpropertiesParent.axesLength[0], coordinates[1] / RFpropertiesParent.axesLength[1], coordinates[2] / RFpropertiesParent.axesLength[2])
	return coordinatesTransformed

def calculateEndCoordinatesPosition2D(neuronComponent):
	endCoordinatesRelativeToCentreCoordinates = ATORpt_RFoperations.calculateRelativePosition2D(neuronComponent.RFproperties.angle, neuronComponent.RFproperties.axesLength[0])
	endCoordinates = (neuronComponent.RFproperties.centerCoordinates[0] + endCoordinatesRelativeToCentreCoordinates[0], neuronComponent.RFproperties.centerCoordinates[1] + endCoordinatesRelativeToCentreCoordinates[1])
	return endCoordinates

def calculateEndCoordinatesPosition3D(neuronComponent):
	endCoordinatesRelativeToCentreCoordinates = ATORpt_RFoperations.calculateRelativePosition3D(neuronComponent.RFproperties.angle, neuronComponent.RFproperties.axesLength)
	endCoordinates = (neuronComponent.RFproperties.centerCoordinates[0] + endCoordinatesRelativeToCentreCoordinates[0], neuronComponent.RFproperties.centerCoordinates[1] + endCoordinatesRelativeToCentreCoordinates[1], neuronComponent.RFproperties.centerCoordinates[2] + endCoordinatesRelativeToCentreCoordinates[2])
	return endCoordinates

def saveRFFilterImage(RFfilter, RFfilterImageFilename):
	RFfilterMask = pt.ne(RFfilter, 0.0).type(pt.float32)
	RFfilterImage = RFfilter + 3.0
	RFfilterImage = RFfilterImage / 4.0
	RFfilterImage = RFfilterImage * RFfilterMask
	RFfilterImage = RFfilterImage * ATORpt_RFoperations.rgbMaxValue
	RFfilterUint8 = RFfilterImage.type(pt.uint8)
	RFfilterNP = RFfilterUint8.numpy()
	ATORpt_RFoperations.saveImage(RFfilterImageFilename, RFfilterNP)

def getFilterDimensions(resolutionProperties, maximumAxisLengthMultiplier=maximumAxisLengthMultiplierDefault, receptiveFieldOpponencyAreaFactor=receptiveFieldOpponencyAreaFactorEllipse):
	axesLengthMax1 = minimumEllipseAxisLength * maximumAxisLengthMultiplier	#1*4
	axesLengthMax2 = minimumEllipseAxisLength * maximumAxisLengthMultiplier	#1*4
	filterRadius = int(max(axesLengthMax1 * receptiveFieldOpponencyAreaFactor, axesLengthMax2 * receptiveFieldOpponencyAreaFactor))	#4*2,4*2
	filterSize = (int(filterRadius * 2), int(filterRadius * 2))	#8*2,8*2
	axesLengthMax = (axesLengthMax1, axesLengthMax2)
	return axesLengthMax, filterRadius, filterSize
