"""ATORpt_RFtri.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt RF Tri - generate tri (as represented by 3 feature points) receptive fields

"""

import torch as pt
import numpy as np
import cv2
import copy

from ATORpt_RFglobalDefs import *
import ATORpt_RFproperties
import ATORpt_RFoperations



def printTriProperties(triProperties):
	print("vertexCoordinatesRelative = ", triProperties.vertexCoordinatesRelative)


def normaliseGlobalTriProperties(ellipseProperties, resolutionFactor):
	resolutionFactor = ellipseProperties.resolutionFactor
	ellipsePropertiesNormalised = copy.deepcopy(ellipseProperties)
	ellipsePropertiesNormalised.centerCoordinates = (ellipsePropertiesNormalised.centerCoordinates[0] * resolutionFactor, ellipsePropertiesNormalised.centerCoordinates[1] * resolutionFactor)
	ellipsePropertiesNormalised.axesLength = (ellipsePropertiesNormalised.axesLength[0] * resolutionFactor, ellipsePropertiesNormalised.axesLength[1] * resolutionFactor)
	return ellipsePropertiesNormalised


def normaliseLocalTriProperties(RFproperties):
	RFpropertiesNormalised = copy.deepcopy(RFproperties)
	RFpropertiesNormalised.angle = triNormalisedAngle
	RFpropertiesNormalised.centerCoordinates = (triNormalisedCentreCoordinates, triNormalisedCentreCoordinates, triNormalisedCentreCoordinates)
	if triRFnormaliseLocalEquilateralTriangle:
		RFpropertiesNormalised.axesLength = ATORpt_RFoperations.getEquilateralTriangleAxesLength(triNormalisedAxesLength)
	else:
		RFpropertiesNormalised.axesLength = (triNormalisedAxesLength, triNormalisedAxesLength)

	return RFpropertiesNormalised


def calculateFilterPixels(filterSize, numberOfDimensions):
	internalFilterSize = getInternalFilterSize(filterSize, numberOfDimensions)

	if numberOfDimensions == 2:
		numberOfFilterPixels = internalFilterSize[0] * internalFilterSize[1] * 3
	elif numberOfDimensions == 3:
		numberOfFilterPixels = internalFilterSize[0] * internalFilterSize[1] * 3

	return numberOfFilterPixels


def getInternalFilterSize(filterSize, numberOfDimensions):
	if numberOfDimensions == 2:
		internalFilterSize = (int(pointFeatureRFinsideRadius * pointFeatureRFopponencyAreaFactor), int(pointFeatureRFinsideRadius * pointFeatureRFopponencyAreaFactor))
	elif numberOfDimensions == 3:
		internalFilterSize = (int(pointFeatureRFinsideRadius * pointFeatureRFopponencyAreaFactor), int(pointFeatureRFinsideRadius * pointFeatureRFopponencyAreaFactor))
	return internalFilterSize


def generateRFfiltersTri(resolutionProperties):
	RFfiltersList = []
	RFfiltersPropertiesList = []
	
	filterTypeIndex = 0

	colourH = (255, 255, 255)
	colourL = (-255, -255, -255)
	RFfiltersHL, RFpropertiesHL = generateRotationalInvariantRFfilters(resolutionProperties, False, colourH, colourL, filterTypeIndex)
	filterTypeIndex += 1

	RFfiltersList.append(RFfiltersHL)
	RFfiltersPropertiesList.append(RFpropertiesHL)

	return RFfiltersList, RFfiltersPropertiesList

def generateRotationalInvariantRFfilters(resolutionProperties, isColourFilter, filterInsideColour, filterOutsideColour, filterTypeIndex):
	RFfiltersList2 = []
	RFfiltersPropertiesList2 = []

	axesLengthMax, filterRadius, filterSize = getFilterDimensions(resolutionProperties)

	for axesLength1 in range(triMinimumAxisLength1, axesLengthMax[0] + 1, triAxesLengthResolution):
		for axesLength2 in range(triMinimumAxisLength2, axesLengthMax[1] + 1, triAxesLengthResolution):
			if axesLength1 > axesLength2:
				if axesLength1 < filterRadius and axesLength2 < filterRadius:
					for angle in range(0, 360, triAngleResolution):
						for corner1OpponencyPosition1 in range(triMinimumCornerOpponencyPosition, triMaximumCornerOpponencyPosition + 1, triCornerOpponencyPositionResolution):
							for corner1OpponencyPosition2 in range(triMinimumCornerOpponencyPosition, triMaximumCornerOpponencyPosition + 1, triCornerOpponencyPositionResolution):
								for corner2OpponencyPosition1 in range(triMinimumCornerOpponencyPosition, triMaximumCornerOpponencyPosition + 1, triCornerOpponencyPositionResolution):
									for corner2OpponencyPosition2 in range(triMinimumCornerOpponencyPosition, triMaximumCornerOpponencyPosition + 1, triCornerOpponencyPositionResolution):
										for corner3OpponencyPosition1 in range(triMinimumCornerOpponencyPosition, triMaximumCornerOpponencyPosition + 1, triCornerOpponencyPositionResolution):
											for corner3OpponencyPosition2 in range(triMinimumCornerOpponencyPosition, triMaximumCornerOpponencyPosition + 1, triCornerOpponencyPositionResolution):
												axesLength = (axesLength1, axesLength2)

												axesLengthInside = (pointFeatureRFinsideRadius, pointFeatureRFinsideRadius)
												axesLengthOutside = (int(pointFeatureRFinsideRadius * pointFeatureRFopponencyAreaFactor), int(pointFeatureRFinsideRadius * pointFeatureRFopponencyAreaFactor))

												angleInside = 0.0
												angleOutside = 0.0
												filterCenterCoordinates = (0, 0)

												RFpropertiesInside = ATORpt_RFproperties.RFpropertiesClass(resolutionProperties.resolutionIndex, resolutionProperties.resolutionFactor, filterSize, ATORpt_RFproperties.RFtypeEllipse, filterCenterCoordinates, axesLengthInside, angleInside, filterInsideColour)
												RFpropertiesOutside = ATORpt_RFproperties.RFpropertiesClass(resolutionProperties.resolutionIndex, resolutionProperties.resolutionFactor, filterSize, ATORpt_RFproperties.RFtypeEllipse, filterCenterCoordinates, axesLengthOutside, angleOutside, filterOutsideColour)
												RFpropertiesInside.isColourFilter = isColourFilter
												RFpropertiesOutside.isColourFilter = isColourFilter

												vertexCoordinatesRelative = ATORpt_RFproperties.deriveTriVertexCoordinatesFromArtificialEllipseProperties(
													axesLength, angle)

												vertexCoordinatesRelativeInside = copy.deepcopy(vertexCoordinatesRelative)
												vertexCoordinatesRelativeOutside = copy.deepcopy(vertexCoordinatesRelative)
												vertexCoordinatesRelativeOutside[0][0] += corner1OpponencyPosition1
												vertexCoordinatesRelativeOutside[0][1] += corner1OpponencyPosition2
												vertexCoordinatesRelativeOutside[1][0] += corner2OpponencyPosition1
												vertexCoordinatesRelativeOutside[1][1] += corner2OpponencyPosition2
												vertexCoordinatesRelativeOutside[2][0] += corner3OpponencyPosition1
												vertexCoordinatesRelativeOutside[2][1] += corner3OpponencyPosition2

												RFfilter = generateRFfilter(resolutionProperties.resolutionIndex, isColourFilter, RFpropertiesInside, RFpropertiesOutside, vertexCoordinatesRelativeInside, vertexCoordinatesRelativeOutside)
												RFfiltersList2.append(RFfilter)

												RFproperties = ATORpt_RFproperties.RFpropertiesClass(resolutionProperties.resolutionIndex, resolutionProperties.resolutionFactor, filterSize, ATORpt_RFproperties.RFtypeTri, filterCenterCoordinates, axesLength, angle, filterInsideColour)
												RFproperties.isColourFilter = isColourFilter
												RFfiltersPropertiesList2.append(RFproperties)

												RFfilterImageFilename = "RFfilterResolutionIndex" + str(resolutionProperties.resolutionIndex) + "filterTypeIndex" + str(filterTypeIndex) + "axesLength1" + str(axesLength1) + "axesLength2" + str(axesLength2) + "angle" + str(angle) + "corner1OpponencyPosition1" + str(corner1OpponencyPosition1) + "corner1OpponencyPosition2" + str(corner1OpponencyPosition2) + "corner2OpponencyPosition1" + str(corner2OpponencyPosition1) + "corner2OpponencyPosition2" + str(corner2OpponencyPosition2) + "corner3OpponencyPosition1" + str(corner3OpponencyPosition1) + "corner3OpponencyPosition2" + str(corner3OpponencyPosition2) + ".png"
												ATORpt_RFproperties.saveRFFilterImage(RFfilter, RFfilterImageFilename)

	RFfiltersTensor = pt.stack(RFfiltersList2, dim=0)

	return RFfiltersTensor, RFfiltersPropertiesList2


def generateRFfilter(resolutionIndex, isColourFilter, RFpropertiesInside, RFpropertiesOutside, vertexCoordinatesRelativeInside, vertexCoordinatesRelativeOutside):

	blankArray = np.full((RFpropertiesInside.imageSize[1], RFpropertiesInside.imageSize[0], rgbNumChannels), 0, np.uint8)
	RFfilterTF = pt.tensor(blankArray, dtype=pt.float32)

	RFpropertiesInside1 = copy.deepcopy(RFpropertiesInside)
	RFpropertiesOutside1 = copy.deepcopy(RFpropertiesOutside)
	RFpropertiesInside2 = copy.deepcopy(RFpropertiesInside)
	RFpropertiesOutside2 = copy.deepcopy(RFpropertiesOutside)
	RFpropertiesInside3 = copy.deepcopy(RFpropertiesInside)
	RFpropertiesOutside3 = copy.deepcopy(RFpropertiesOutside)

	RFpropertiesInside1.centerCoordinates = (RFpropertiesInside1.centerCoordinates[0] + vertexCoordinatesRelativeInside[0][0], RFpropertiesInside1.centerCoordinates[1] + vertexCoordinatesRelativeInside[0][1])
	RFpropertiesOutside1.centerCoordinates = (RFpropertiesOutside1.centerCoordinates[0] + vertexCoordinatesRelativeOutside[0][0], RFpropertiesOutside1.centerCoordinates[1] + vertexCoordinatesRelativeOutside[0][1])
	RFpropertiesInside2.centerCoordinates = (RFpropertiesInside2.centerCoordinates[0] + vertexCoordinatesRelativeInside[1][0], RFpropertiesInside2.centerCoordinates[1] + vertexCoordinatesRelativeInside[1][1])
	RFpropertiesOutside2.centerCoordinates = (RFpropertiesOutside2.centerCoordinates[0] + vertexCoordinatesRelativeOutside[1][0], RFpropertiesOutside2.centerCoordinates[1] + vertexCoordinatesRelativeOutside[1][1])
	RFpropertiesInside3.centerCoordinates = (RFpropertiesInside3.centerCoordinates[0] + vertexCoordinatesRelativeInside[2][0], RFpropertiesInside3.centerCoordinates[1] + vertexCoordinatesRelativeInside[2][1])
	RFpropertiesOutside3.centerCoordinates = (RFpropertiesOutside3.centerCoordinates[0] + vertexCoordinatesRelativeOutside[2][0], RFpropertiesOutside3.centerCoordinates[1] + vertexCoordinatesRelativeOutside[2][1])

	if generatePointFeatureCorners:
		drawFeatureType = ATORpt_RFproperties.RFfeatureTypeCorner
	else:
		drawFeatureType = ATORpt_RFproperties.RFfeatureTypePoint

	RFfilterTF = ATORpt_RFproperties.drawRF(RFfilterTF, RFpropertiesInside1, RFpropertiesOutside1, drawFeatureType, True)
	RFfilterTF = ATORpt_RFproperties.drawRF(RFfilterTF, RFpropertiesInside2, RFpropertiesOutside2, drawFeatureType, True)
	RFfilterTF = ATORpt_RFproperties.drawRF(RFfilterTF, RFpropertiesInside3, RFpropertiesOutside3, drawFeatureType, True)

	if ATORpt_RFoperations.storeRFfiltersValuesAsFractions:
		RFfilterTF = RFfilterTF / ATORpt_RFoperations.rgbMaxValue

	if not isColourFilter:
		RFfilterTF = pt.mean(RFfilterTF, dim=2, keepdim=True)

	return RFfilterTF

def getFilterDimensions(resolutionProperties):
	return ATORpt_RFproperties.getFilterDimensions(resolutionProperties, triMaximumAxisLengthMultiplier, receptiveFieldOpponencyAreaFactorTri)
