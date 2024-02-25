"""ATORpt_RFellipse.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt RF Ellipse - generate ellipse receptive fields

"""

import torch as pt
import numpy as np
import cv2
import copy

import ATORpt_RFproperties
import ATORpt_RFellipseProperties
import ATORpt_RFoperations

ellipseAngleResolution = 10  # degrees
minimumEllipseFitErrorRequirement = 1500.0  # calibrate

# match ATORpt_RFellipse algorithm;
RFnormaliseLocalEquilateralTriangle = True

ellipseNormalisedAngle = 0.0
ellipseNormalisedCentreCoordinates = 0.0
ellipseNormalisedAxesLength = 1.0

# ellipse axesLength definition (based on https://docs.opencv.org/4.5.3/d6/d6e/group__imgproc__draw.html#ga28b2267d35786f5f890ca167236cbc69)
#  ___
# / | \ minimumEllipseAxisLength1 | (axis1: axes width)
# | --| minimumEllipseAxisLength2 - (axis2: axis height)
# \___/
#
minimumEllipseAxisLength1 = ATORpt_RFproperties.minimumEllipseAxisLength * 2  # minimum elongation is required	# can be set to 1 as (axesLength1 > axesLength2) condition is enforced for RFellipse creation
minimumEllipseAxisLength2 = ATORpt_RFproperties.minimumEllipseAxisLength
if ATORpt_RFproperties.lowResultFilterPosition:
	ellipseAxesLengthResolution = 2  # pixels (at resolution r)	# use course grain resolution to decrease number of filters # OLD: 1
else:
	ellipseAxesLengthResolution = 1  # pixels (at resolution r)
ellipseAngleResolution = 30  # degrees
ellipseColourResolution = 64  # bits


def normaliseGlobalEllipseProperties(ellipseProperties, resolutionFactor):
	return ATORpt_RFellipseProperties.normaliseGlobalEllipseProperties(ellipseProperties, resolutionFactor)


def normaliseLocalEllipseProperties(RFproperties):
	# normalise ellipse respect to major/minor ellipticity axis orientation (WRT self)
	RFpropertiesNormalised = copy.deepcopy(RFproperties)
	RFpropertiesNormalised.angle = ellipseNormalisedAngle  # CHECKTHIS
	RFpropertiesNormalised.centerCoordinates = (
	ellipseNormalisedCentreCoordinates, ellipseNormalisedCentreCoordinates, ellipseNormalisedCentreCoordinates)
	if RFnormaliseLocalEquilateralTriangle:
		RFpropertiesNormalised.axesLength = ATORpt_RFoperations.getEquilateralTriangleAxesLength(
			ellipseNormalisedAxesLength)
	else:
		RFpropertiesNormalised.axesLength = (ellipseNormalisedAxesLength, ellipseNormalisedAxesLength)
	return RFpropertiesNormalised


def calculateFilterPixels(filterSize, numberOfDimensions):
	internalFilterSize = getInternalFilterSize(filterSize, numberOfDimensions)  # CHECKTHIS: only consider contribution of positive (additive) pixels
	# print("internalFilterSize = ", internalFilterSize)

	if numberOfDimensions == 2:
		numberOfFilterPixels = internalFilterSize[0] * internalFilterSize[1]
	elif numberOfDimensions == 3:
		numberOfFilterPixels = internalFilterSize[0] * internalFilterSize[1]  # CHECKTHIS
	# print("numberOfFilterPixels = ", numberOfFilterPixels)
	return numberOfFilterPixels


def getInternalFilterSize(filterSize, numberOfDimensions):
	if numberOfDimensions == 2:
		internalFilterSize = (
		(filterSize[0] / ATORpt_RFproperties.receptiveFieldOpponencyAreaFactorEllipse),
		(filterSize[1] / ATORpt_RFproperties.receptiveFieldOpponencyAreaFactorEllipse))
	elif numberOfDimensions == 3:
		internalFilterSize = (
		(filterSize[0] / ATORpt_RFproperties.receptiveFieldOpponencyAreaFactorEllipse),
		(filterSize[1] / ATORpt_RFproperties.receptiveFieldOpponencyAreaFactorEllipse))  # CHECKTHIS
	return internalFilterSize


def generateRFfiltersEllipse(resolutionProperties, RFfiltersList, RFfiltersPropertiesList):
	# generate filter types

	# 2D code;

	# filters are generated based on human magnocellular/parvocellular/koniocellular wavelength discrimination in LGN and VX (double/opponent receptive fields)
	filterTypeIndex = 0

	# magnocellular filters (monochromatic);
	colourH = (255, 255, 255)  # high
	colourL = (-255, -255, -255)  # low
	RFfiltersHL, RFpropertiesHL = generateRotationalInvariantRFfilters(resolutionProperties, False, colourH, colourL,
																	  filterTypeIndex)
	filterTypeIndex += 1
	RFfiltersLH, RFpropertiesLH = generateRotationalInvariantRFfilters(resolutionProperties, False, colourL, colourH,
																	  filterTypeIndex)
	filterTypeIndex += 1

	# parvocellular/koniocellular filters (based on 2 cardinal colour axes; ~red-~green, ~blue-~yellow);
	colourRmG = (255, -255, 0)  # red+, green-
	colourGmR = (-255, 255, 0)  # green+, red-
	colourBmY = (-127, -127, 255)  # blue+, yellow-
	colourYmB = (127, 127, -255)  # yellow+, blue-
	RFfiltersRG, RFpropertiesRG = generateRotationalInvariantRFfilters(resolutionProperties, True, colourRmG, colourGmR,
																	 filterTypeIndex)
	filterTypeIndex += 1
	RFfiltersGR, RFpropertiesGR = generateRotationalInvariantRFfilters(resolutionProperties, True, colourGmR, colourRmG,
																	 filterTypeIndex)
	filterTypeIndex += 1
	RFfiltersBY, RFpropertiesBY = generateRotationalInvariantRFfilters(resolutionProperties, True, colourBmY, colourYmB,
																	 filterTypeIndex)
	filterTypeIndex += 1
	RFfiltersYB, RFpropertiesYB = generateRotationalInvariantRFfilters(resolutionProperties, True, colourYmB, colourBmY,
																	 filterTypeIndex)
	filterTypeIndex += 1

	RFfiltersList.append(RFfiltersHL)
	RFfiltersList.append(RFfiltersLH)
	RFfiltersList.append(RFfiltersRG)
	RFfiltersList.append(RFfiltersGR)
	RFfiltersList.append(RFfiltersBY)
	RFfiltersList.append(RFfiltersYB)

	RFfiltersPropertiesList.append(RFpropertiesHL)
	RFfiltersPropertiesList.append(RFpropertiesLH)
	RFfiltersPropertiesList.append(RFpropertiesRG)
	RFfiltersPropertiesList.append(RFpropertiesGR)
	RFfiltersPropertiesList.append(RFpropertiesBY)
	RFfiltersPropertiesList.append(RFpropertiesYB)


def generateRotationalInvariantRFfilters(resolutionProperties, isColourFilter, filterInsideColour, filterOutsideColour,
										 filterTypeIndex):

	RFfiltersList2 = []
	RFfiltersPropertiesList2 = []

	# FUTURE: consider storing filters in n dimensional array and finding local minima of filter matches across all dimensions

	# reduce max size of ellipse at each res
	axesLengthMax, filterRadius, filterSize = ATORpt_RFproperties.getFilterDimensions(resolutionProperties)

	# print("axesLengthMax = ", axesLengthMax)

	for axesLength1 in range(minimumEllipseAxisLength1, axesLengthMax[0] + 1, ellipseAxesLengthResolution):
		for axesLength2 in range(minimumEllipseAxisLength2, axesLengthMax[1] + 1, ellipseAxesLengthResolution):
			if axesLength1 > axesLength2:  # ensure that ellipse is always alongated towards axis1 (required for consistent normalisation)
				for angle in range(0, 360, ellipseAngleResolution):  # degrees

					axesLengthInside = (axesLength1, axesLength2)
					axesLengthOutside = (int(axesLength1 * ATORpt_RFproperties.receptiveFieldOpponencyAreaFactorEllipse),
										 int(axesLength2 * ATORpt_RFproperties.receptiveFieldOpponencyAreaFactorEllipse))
					filterCenterCoordinates = (0, 0)

					RFpropertiesInside = ATORpt_RFproperties.RFpropertiesClass(resolutionProperties.resolutionIndex,
																			   resolutionProperties.resolutionFactor,
																			   filterSize,
																			   ATORpt_RFproperties.RFtypeEllipse,
																			   filterCenterCoordinates, axesLengthInside,
																			   angle, filterInsideColour)
					RFpropertiesOutside = ATORpt_RFproperties.RFpropertiesClass(resolutionProperties.resolutionIndex,
																				resolutionProperties.resolutionFactor,
																				filterSize,
																				ATORpt_RFproperties.RFtypeEllipse,
																				filterCenterCoordinates,
																				axesLengthOutside, angle,
																				filterOutsideColour)
					RFpropertiesInside.isColourFilter = isColourFilter
					RFpropertiesOutside.isColourFilter = isColourFilter

					RFfilter = generateRFfilter(resolutionProperties.resolutionIndex, isColourFilter, RFpropertiesInside,
												RFpropertiesOutside)
					RFfiltersList2.append(RFfilter)

					RFproperties = copy.deepcopy(RFpropertiesInside)
					# RFproperties.centerCoordinates = centerCoordinates	# centerCoordinates are set after filter is applied to imageSegment
					RFfiltersPropertiesList2.append(
						RFproperties)  # CHECKTHIS: use RFpropertiesInside not RFpropertiesOutside

					# debug:
					# print(RFfilter.shape)
					if resolutionProperties.debugVerbose:
						ATORpt_RFproperties.printRFproperties(RFproperties)
						# ATORpt_RFproperties.printRFproperties(RFpropertiesInside)
						# ATORpt_RFproperties.printRFproperties(RFpropertiesOutside)
					# print("RFfilter = ", RFfilter)

					RFfilterImageFilename = "RFfilterResolutionIndex" + str(
						resolutionProperties.resolutionIndex) + "filterTypeIndex" + str(
						filterTypeIndex) + "axesLength1" + str(axesLength1) + "axesLength2" + str(
						axesLength2) + "angle" + str(angle) + ".png"
					ATORpt_RFproperties.saveRFFilterImage(RFfilter, RFfilterImageFilename)

	# create 3D tensor (for hardware accelerated test/application of filters)
	RFfiltersTensor = pt.stack(RFfiltersList2, dim=0)

	return RFfiltersTensor, RFfiltersPropertiesList2


def generateRFfilter(resolutionProperties, isColourFilter, RFpropertiesInside, RFpropertiesOutside):

	# RF filter example (RFfilterTF):
	#
	# 0 0 0 0 0 0
	# 0 0 - - 0 0
	# 0 - + + - 0
	# 0 0 - - 0 0
	# 0 0 0 0 0 0
	#
	# where "-" = -RFcolourOutside [R G B], "+" = +RFcolourInside [R G B], and "0" = [0, 0, 0]

	# generate ellipse on blank canvas
	blankArray = np.full((RFpropertiesInside.imageSize[1], RFpropertiesInside.imageSize[0], ATORpt_RFoperations.rgbNumChannels), 0, np.uint8)  # rgb
	RFfilterTF = pt.tensor(blankArray, dtype=pt.float32)


	RFfilterTF = ATORpt_RFproperties.drawRF(RFfilterTF, RFpropertiesInside, RFpropertiesOutside,
											ATORpt_RFproperties.RFfeatureTypeEllipse, False)

	# print("RFfilterTF = ", RFfilterTF)

	if ATORpt_RFoperations.storeRFfiltersValuesAsFractions:
		RFfilterTF = RFfilterTF / ATORpt_RFoperations.rgbMaxValue

	if not isColourFilter:
		# RFfilterTF = tf.image.rgb_to_grayscale(RFfilterTF)
		RFfilterTF = RFfilterTF.mean(dim=-1, keepdim=True)

	# print("RFfilterTF.shape = ", RFfilterTF.shape)
	# print("RFfilterTF = ", RFfilterTF)

	return RFfilterTF

		

