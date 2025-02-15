"""ATORpt_RFapplyFilter.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORpt_RFmainFT.py

# Usage:
See ATORpt_RFmainFT.py

# Description:
ATORpt RF Filter - RF Filter transformations (pixel space)

"""

import torch as pt
import torch.nn.functional as F
import numpy as np
import copy

from ATORpt_RFglobalDefs import *
import ATORpt_pta_image as pta_image
import ATORpt_RFpropertiesClass
import ATORpt_RFgenerateEllipse
import ATORpt_RFgenerateTri
import ATORpt_RFoperations


def calculateFilterApplicationResultThreshold(filterApplicationResult, minimumFilterRequirement, filterSize, isColourFilter, numberOfDimensions, RFtype):
	minimumFilterRequirementLocal = minimumFilterRequirement * calculateFilterPixels(filterSize, numberOfDimensions, RFtype)

	if(RFuseParallelProcessedCNN):
		pass
		#TODO: minimumFilterRequirementLocal REQUIRES CALIBRATION based on CNN operation
	else:
		# if(isColourFilter):
		# 	minimumFilterRequirementLocal = minimumFilterRequirementLocal*rgbNumChannels*rgbNumChannels  # CHECKTHIS  # not required as assume filter colours will be normalised to the maximum value of a single rgb channel?
		if not ATORpt_RFoperations.storeRFfiltersValuesAsFractions:
			minimumFilterRequirementLocal = minimumFilterRequirementLocal * (ATORpt_RFoperations.rgbMaxValue * ATORpt_RFoperations.rgbMaxValue)  # rgbMaxValue of both imageSegment and RFfilter

	print("minimumFilterRequirementLocal = ", minimumFilterRequirementLocal)
	print("pt.max(filterApplicationResult) = ", pt.max(filterApplicationResult))

	filterApplicationResultThreshold = filterApplicationResult > minimumFilterRequirementLocal
	return filterApplicationResultThreshold


def calculateFilterPixels(filterSize, numberOfDimensions, RFtype):
	if RFtype == RFtypeEllipse:
		return ATORpt_RFgenerateEllipse.calculateFilterPixels(filterSize, numberOfDimensions)
	elif RFtype == RFtypeTri:
		return ATORpt_RFgenerateTri.calculateFilterPixels(filterSize, numberOfDimensions)	#CHECKTHIS
	elif RFtype == RFtypeTemporaryPointFeatureKernel:
		return ATORpt_RFgenerateTri.calculateFilterPixels(filterSize, numberOfDimensions)	#CHECKTHIS

def normaliseRFfilter(RFfilter, RFproperties):
	# normalise ellipse respect to major/minor ellipticity axis orientation (WRT self)
	RFfilterNormalised = transformRFfilter(RFfilter, RFproperties)
	# RFfilterNormalised = RFfilter
	return RFfilterNormalised


def transformRFfilter(RFfilter, RFpropertiesParent):
	if RFpropertiesParent.numberOfDimensions == 2:
		centerCoordinates = [-RFpropertiesParent.centerCoordinates[0], -RFpropertiesParent.centerCoordinates[1]]
		print("centerCoordinates = ", centerCoordinates)
		axesLength = 1.0 / RFpropertiesParent.axesLength[0]  # [1.0/RFpropertiesParent.axesLength[0], 1.0/RFpropertiesParent.axesLength[1]]
		angle = -RFpropertiesParent.angle
		RFfilterTransformed = transformRFfilter2D(RFfilter, centerCoordinates, axesLength, angle)
	elif RFpropertiesParent.numberOfDimensions == 3:
		print("error transformRFfilterWRTparent: RFpropertiesParent.numberOfDimensions == 3 not yet coded")
		quit()
	return RFfilterTransformed


def transformRFfilter2D(RFfilter, centerCoordinates, axesLength, angle):
	# CHECKTHIS: 2D code only;
	RFfilter = RFfilter.permute(2, 0, 1)	#ensure channels dim is first
	#RFfilterTransformed = pt.unsqueeze(RFfilter, 0)  # add batch dim
	RFfilterTransformed = RFfilter
	angleRadians = ATORpt_RFoperations.convertDegreesToRadians(angle)
	#print("RFfilterTransformed.shape = ", RFfilterTransformed.shape)
	RFfilterTransformed = pta_image.rotate(RFfilterTransformed, angleRadians, fillValue=RFfilterImageTransformFillValue)
	if(pta_image_rotate_doesNotSupportCUDA):
		RFfilterTransformed = RFfilterTransformed.to(device)
	centerCoordinatesList = [float(x) for x in list(centerCoordinates)]
	RFfilterTransformed = pta_image.translate(RFfilterTransformed, centerCoordinatesList, fillValue=RFfilterImageTransformFillValue)
	# print("axesLength = ", axesLength)
	#print("RFfilterTransformed.shape = ", RFfilterTransformed.shape)
	RFfilterTransformed = pta_image.scale(RFfilterTransformed, axesLength, fillValue=RFfilterImageTransformFillValue)
	#print("RFfilterTransformed.shape = ", RFfilterTransformed.shape)
	RFfilterTransformed = pt.squeeze(RFfilterTransformed)
	RFfilter = RFfilter.permute(1, 2, 0)	#ensure channels dim is last
	return RFfilterTransformed

def rotateRFfilter(RFfilter, RFproperties):
	return rotateRFfilter(-RFproperties.angle)


def rotateRFfilter(RFfilter, angle):
	RFfilter = pt.unsqueeze(RFfilter, 0)  # add extra dimension for num_images
	return RFfilterNormalised


def getFilterDimensions(resolutionProperties):
	return ATORpt_RFpropertiesClass.getFilterDimensions(resolutionProperties)


# CHECKTHIS: upgrade code to support ATORpt_RFgenerateTri
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

