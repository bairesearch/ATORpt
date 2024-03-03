"""ATORpt_RFproperties.py

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
import numpy as np
import cv2
import copy

from ATORpt_RFglobalDefs import *
import ATORpt_RFellipseProperties
import ATORpt_RFoperations


class RFpropertiesClass(ATORpt_RFellipseProperties.EllipsePropertiesClass):
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
		ATORpt_RFellipseProperties.printEllipseProperties(RFproperties)
	elif RFproperties.RFtype == RFtypeTri:
		ATORpt_RFellipseProperties.printEllipseProperties(RFproperties)
		print("vertexCoordinatesRelative = ", RFproperties.vertexCoordinatesRelative)

def drawRF(RFfilterTF, RFpropertiesInside, RFpropertiesOutside, drawFeatureType, drawFeatureOverlay):
	blankArray = np.full((RFpropertiesInside.imageSize[1], RFpropertiesInside.imageSize[0], 1), 0, np.uint8)
	ellipseFilterImageInside = copy.deepcopy(blankArray)
	ellipseFilterImageOutside = copy.deepcopy(blankArray)
	RFpropertiesInsideWhite = copy.deepcopy(RFpropertiesInside)
	RFpropertiesInsideWhite.colour = (255, 255, 255)
	RFpropertiesOutsideWhite = copy.deepcopy(RFpropertiesOutside)
	RFpropertiesOutsideWhite.colour = (255, 255, 255)
	RFpropertiesInsideBlack = copy.deepcopy(RFpropertiesInside)
	RFpropertiesInsideBlack.colour = (000, 000, 000)
	if drawFeatureType == RFfeatureTypeEllipse:
		ATORpt_RFellipseProperties.drawEllipse(ellipseFilterImageInside, RFpropertiesInsideWhite, True)
		ATORpt_RFellipseProperties.drawEllipse(ellipseFilterImageOutside, RFpropertiesOutsideWhite, True)
		ATORpt_RFellipseProperties.drawEllipse(ellipseFilterImageOutside, RFpropertiesInsideBlack, True)
	elif drawFeatureType == RFfeatureTypeCircle:
		ATORpt_RFellipseProperties.drawCircle(ellipseFilterImageInside, RFpropertiesInsideWhite, True)
		ATORpt_RFellipseProperties.drawCircle(ellipseFilterImageOutside, RFpropertiesOutsideWhite, True)
		ATORpt_RFellipseProperties.drawCircle(ellipseFilterImageOutside, RFpropertiesInsideBlack, True)
	elif drawFeatureType == RFfeatureTypePoint:
		ATORpt_RFellipseProperties.drawPoint(ellipseFilterImageInside, RFpropertiesInsideWhite, True)
		ATORpt_RFellipseProperties.drawCircle(ellipseFilterImageOutside, RFpropertiesOutsideWhite, True)
		ATORpt_RFellipseProperties.drawPoint(ellipseFilterImageOutside, RFpropertiesInsideBlack, True)
	elif drawFeatureType == RFfeatureTypeCorner:
		ATORpt_RFellipseProperties.drawPoint(ellipseFilterImageInside, RFpropertiesInsideWhite, True)
		ATORpt_RFellipseProperties.drawRectangle(ellipseFilterImageOutside, RFpropertiesOutsideWhite, True)
		ATORpt_RFellipseProperties.drawPoint(ellipseFilterImageOutside, RFpropertiesInsideBlack, True)
	insideImageTF = pt.tensor(ellipseFilterImageInside, dtype=pt.float32)
	insideImageTF = pt.gt(insideImageTF, 0.0)
	insideImageTF = insideImageTF.type(pt.float32)
	outsideImageTF = pt.tensor(ellipseFilterImageOutside, dtype=pt.float32)
	outsideImageTF = pt.gt(outsideImageTF, 0.0)
	outsideImageTF = outsideImageTF.type(pt.float32)
	multiples = tuple([1, 1, 3])
	insideImageTF = insideImageTF.repeat(multiples)
	RFcolourInside = pt.tensor([RFpropertiesInside.colour[0], RFpropertiesInside.colour[1], RFpropertiesInside.colour[2]], dtype=pt.float32)
	RFcolourInside = RFcolourInside.unsqueeze(dim=0)
	insideImageTF = insideImageTF * RFcolourInside
	multiples = tuple([1, 1, 3])
	outsideImageTF = outsideImageTF.repeat(multiples)
	RFcolourOutside = pt.tensor([RFpropertiesOutside.colour[0], RFpropertiesOutside.colour[1], RFpropertiesOutside.colour[2]], dtype=pt.float32)
	RFcolourOutside = RFcolourOutside.unsqueeze(dim=0)
	outsideImageTF = outsideImageTF * RFcolourOutside
	if drawFeatureOverlay:
		RFfilterTFon = pt.ne(RFfilterTF, 0.0)
		insideImageTFon = pt.ne(insideImageTF, 0.0)
		outsideImageTFon = pt.ne(outsideImageTF, 0.0)
		insideImageTFonOverlap = pt.logical_and(RFfilterTFon, insideImageTFon)
		outsideImageTFonOverlap = pt.logical_and(RFfilterTFon, outsideImageTFon)
		insideImageTFonMask = pt.logical_not(insideImageTFonOverlap).type(pt.float32)
		outsideImageTFonMask = pt.logical_not(outsideImageTFonOverlap).type(pt.float32)
		insideImageTF = insideImageTF * insideImageTFonMask
		outsideImageTF = outsideImageTF * outsideImageTFonMask
		RFfilterTF = RFfilterTF + insideImageTF + outsideImageTF
	else:
		RFfilterTF = RFfilterTF + insideImageTF + outsideImageTF
	return RFfilterTF

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
