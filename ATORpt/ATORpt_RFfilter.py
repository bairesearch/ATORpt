"""ATORpt_RFfilter.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt RF Filter - RF Filter transformations (pixel space)

"""

import torch as pt
import torch
import torch.nn.functional as F
import numpy as np
import copy

import ATORpt_pta_image as pta_image
import ATORpt_RFproperties
import ATORpt_RFellipse
import ATORpt_RFtri
import ATORpt_RFoperations

minimumFilterRequirement = 1.5  # CHECKTHIS: calibrate  # matched values fraction  # theoretical value: 0.95

# if(debugSaveRFfiltersAndImageSegments):
RFfilterImageTransformFillValue = 0.0


def calculateFilterApplicationResultThreshold(filterApplicationResult, minimumFilterRequirement, filterSize,
											  isColourFilter, numberOfDimensions, RFtype):
	minimumFilterRequirementLocal = minimumFilterRequirement * calculateFilterPixels(filterSize, numberOfDimensions,
																					 RFtype)

	# if(isColourFilter):
	# 	minimumFilterRequirementLocal = minimumFilterRequirementLocal*ATORpt_RFoperations.rgbNumChannels*ATORpt_RFoperations.rgbNumChannels  # CHECKTHIS  # not required as assume filter colours will be normalised to the maximum value of a single rgb channel?
	if not ATORpt_RFoperations.storeRFfiltersValuesAsFractions:
		minimumFilterRequirementLocal = minimumFilterRequirementLocal * (ATORpt_RFoperations.rgbMaxValue * ATORpt_RFoperations.rgbMaxValue)  # rgbMaxValue of both imageSegment and RFfilter

	print("minimumFilterRequirementLocal = ", minimumFilterRequirementLocal)
	print("pt.max(filterApplicationResult) = ", pt.max(filterApplicationResult))

	filterApplicationResultThreshold = filterApplicationResult > minimumFilterRequirementLocal
	return filterApplicationResultThreshold


def calculateFilterPixels(filterSize, numberOfDimensions, RFtype):
	if RFtype == ATORpt_RFproperties.RFtypeEllipse:
		return ATORpt_RFellipse.calculateFilterPixels(filterSize, numberOfDimensions)
	elif RFtype == ATORpt_RFproperties.RFtypeTri:
		return ATORpt_RFtri.calculateFilterPixels(filterSize, numberOfDimensions)


def normaliseRFfilter(RFfilter, RFproperties):
	# normalise ellipse respect to major/minor ellipticity axis orientation (WRT self)
	RFfilterNormalised = transformRFfilterTF(RFfilter, RFproperties)
	# RFfilterNormalised = RFfilter
	return RFfilterNormalised


def transformRFfilterTF(RFfilter, RFpropertiesParent):
	if RFpropertiesParent.numberOfDimensions == 2:
		centerCoordinates = [-RFpropertiesParent.centerCoordinates[0], -RFpropertiesParent.centerCoordinates[1]]
		axesLength = 1.0 / RFpropertiesParent.axesLength[0]  # [1.0/RFpropertiesParent.axesLength[0], 1.0/RFpropertiesParent.axesLength[1]]
		angle = -RFpropertiesParent.angle
		RFfilterTransformed = transformRFfilterTF2D(RFfilter, centerCoordinates, axesLength, angle)
	elif RFpropertiesParent.numberOfDimensions == 3:
		print("error transformRFfilterWRTparentTF: RFpropertiesParent.numberOfDimensions == 3 not yet coded")
		quit()
	return RFfilterTransformed


def transformRFfilterTF2D(RFfilter, centerCoordinates, axesLength, angle):
	# CHECKTHIS: 2D code only;
	# RFfilterTransformed = pt.unsqueeze(RFfilterTransformed, 0)  # add extra dimension for num_images
	RFfilterTransformed = RFfilter
	angleRadians = ATORpt_RFoperations.convertDegreesToRadians(angle)
	RFfilterTransformed = pta_image.rotate(RFfilterTransformed, angleRadians, fill_value=RFfilterImageTransformFillValue)
	centerCoordinatesList = [float(x) for x in list(centerCoordinates)]
	RFfilterTransformed = pta_image.translate(RFfilterTransformed, centerCoordinatesList, fill_value=RFfilterImageTransformFillValue)
	# print("axesLength = ", axesLength)
	# print("RFfilterTransformed.shape = ", RFfilterTransformed.shape)
	RFfilterTransformed = tfa_image.scale(RFfilterTransformed, axesLength, fill_value=RFfilterImageTransformFillValue)
	# print("RFfilterTransformed.shape = ", RFfilterTransformed.shape)
	RFfilterTransformed = pt.squeeze(RFfilterTransformed)
	return RFfilterTransformed

def rotateRFfilterTF(RFfilter, RFproperties):
	return rotateRFfilterTF(-RFproperties.angle)


def rotateRFfilterTF(RFfilter, angle):
	RFfilter = pt.unsqueeze(RFfilter, 0)  # add extra dimension for num_images
	return RFfilterNormalised


def getFilterDimensions(resolutionProperties):
	return ATORpt_RFproperties.getFilterDimensions(resolutionProperties)


# CHECKTHIS: upgrade code to support ATORpt_RFtri
def allFilterCoordinatesWithinImage(centerCoordinates, filterRadius, imageSize):
	imageSegmentStart = (centerCoordinates[0] - filterRadius, centerCoordinates[1] - filterRadius)
	imageSegmentEnd = (centerCoordinates[0] + filterRadius, centerCoordinates[1] + filterRadius)
	if (imageSegmentStart[0] >= 0 and imageSegmentStart[1] >= 0 and imageSegmentEnd[0] < imageSize[0] and
			imageSegmentEnd[1] < imageSize[1]):
		result = True
	else:
		result = False
		# create artificial image segment (will be discarded during image filter application)
		imageSegmentStart = (0, 0)
		imageSegmentEnd = (filterRadius * 2, filterRadius * 2)
	return result, imageSegmentStart, imageSegmentEnd

