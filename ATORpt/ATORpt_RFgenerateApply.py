"""ATORpt_RFgenerateApply.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt RF (receptive field)

"""

import torch as pt
import torch
import torch.nn.functional as F
import numpy as np
import copy
import math

from ATORpt_RFglobalDefs import *
import ATORpt_RFellipseProperties
import ATORpt_RFellipse
import ATORpt_RFtri
import ATORpt_RFfilter
import ATORpt_RFproperties
import ATORpt_RFoperations
if(RFuseParallelProcessedCNN):
	import ATORpt_RFCNN

class RFneuronClass():
	def __init__(self, resolutionProperties, RFproperties, RFfilter, RFImage):
		self.resolutionIndex = resolutionProperties.resolutionIndex
		self.RFproperties = RFproperties
		# if(RFsaveRFfiltersAndImageSegments):
		self.RFfilter = RFfilter
		self.RFneuronParents = []
		self.RFneuronComponents = []


def normaliseRFComponentWRTparent(resolutionProperties, neuronComponent, RFpropertiesParent):
	neuronComponent.RFpropertiesNormalisedWRTparent = ATORpt_RFproperties.generateRFtransformedProperties(neuronComponent, RFpropertiesParent)
	if RFsaveRFfiltersAndImageSegments:
		neuronComponent.RFfilterNormalisedWRTparent = ATORpt_RFfilter.transformRFfilterTF(neuronComponent.RFfilter, RFpropertiesParent)
		if(RFsaveRFimageSegments):
			neuronComponent.RFImageNormalisedWRTparent = ATORpt_RFfilter.transformRFfilterTF(neuronComponent.RFImage, RFpropertiesParent)


def childRFoverlapsParentRF(neuronRFpropertiesNormalisedGlobal, lowerNeuronRFpropertiesNormalisedGlobal):
	if neuronRFpropertiesNormalisedGlobal.RFtype == RFtypeEllipse:
		return ATORpt_RFellipseProperties.centroidOverlapsEllipse(neuronRFpropertiesNormalisedGlobal, lowerNeuronRFpropertiesNormalisedGlobal)
	elif neuronRFpropertiesNormalisedGlobal.RFtype == RFtypeTri:
		# CHECKTHIS is appropriate for tri;
		return ATORpt_RFellipseProperties.centroidOverlapsEllipse(neuronRFpropertiesNormalisedGlobal, lowerNeuronRFpropertiesNormalisedGlobal)


def normaliseGlobalRFproperties(RFproperties, resolutionFactor):
	# normalise RF respect to original image size
	if RFproperties.RFtype == RFtypeEllipse:
		RFpropertiesNormalisedGlobal = ATORpt_RFellipse.normaliseGlobalEllipseProperties(RFproperties, resolutionFactor)
	elif RFproperties.RFtype == RFtypeTri:
		RFpropertiesNormalisedGlobal = ATORpt_RFtri.normaliseGlobalTriProperties(RFproperties, resolutionFactor)
	return RFpropertiesNormalisedGlobal


def normaliseLocalRFproperties(RFproperties):
	# normalise ellipse respect to major/minor ellipticity axis orientation (WRT self)
	if RFproperties.RFtype == RFtypeEllipse:
		ATORpt_RFellipse.normaliseLocalEllipseProperties(RFproperties)
	elif RFproperties.RFtype == RFtypeTri:
		# CHECKTHIS is appropriate for tri; (ellipseNormalisedAxesLength, ellipseNormalisedAxesLength/q) might create equilateral triangle
		ATORpt_RFtri.normaliseLocalTriProperties(RFproperties)


def generateRFfilters(resolutionProperties, generateRFfiltersEllipse, generateRFfiltersTri):
	RFfiltersFeatureTypeList = []
	RFfiltersPropertiesFeatureTypeList = []
	if generateRFfiltersEllipse:
		RFfiltersList, RFfiltersPropertiesList = ATORpt_RFellipse.generateRFfiltersEllipse(resolutionProperties)
		RFfiltersFeatureTypeList.append(RFfiltersList)
		RFfiltersPropertiesFeatureTypeList.append(RFfiltersPropertiesList)
	if generateRFfiltersTri:
		RFfiltersList, RFfiltersPropertiesList = ATORpt_RFtri.generateRFfiltersTri(resolutionProperties)
		RFfiltersFeatureTypeList.append(RFfiltersList)
		RFfiltersPropertiesFeatureTypeList.append(RFfiltersPropertiesList)
	return RFfiltersFeatureTypeList, RFfiltersPropertiesFeatureTypeList


def applyRFfilters(resolutionProperties, RFfiltersTensor, numberOfDimensions, RFfiltersPropertiesList2, inputImageSegments=None, inputImage=None, RFfiltersConv=None):
	# perform convolution for each filter size;
	axesLengthMax, filterRadius, filterSize = ATORpt_RFfilter.getFilterDimensions(resolutionProperties)
	RFpropertiesList = []
	
	if(RFuseParallelProcessedCNN):
		isColourFilter = RFfiltersPropertiesList2[0].isColourFilter
		numberOfKernels = RFfiltersPropertiesList2[0].numberOfKernels
		#print("isColourFilter = ", isColourFilter)
		#print("numberOfKernels = ", numberOfKernels)
		filterApplicationResult = ATORpt_RFCNN.applyCNNfilters(inputImage, RFfiltersConv, isColourFilter, numberOfKernels)	#dim: numberOfImageSegments*numberOfKernels
	else:
		inputImageSegmentsPixelsFlattened = inputImageSegments.view(inputImageSegments.shape[0], -1)
		RFfiltersTensorPixelsFlattened = RFfiltersTensor.view(RFfiltersTensor.shape[0], -1)
		filterApplicationResult = pt.matmul(inputImageSegmentsPixelsFlattened, RFfiltersTensorPixelsFlattened.t())
		filterApplicationResult = filterApplicationResult.view(-1)	#dim: numberOfImageSegments*numberOfKernels

	isColourFilter = RFfiltersPropertiesList2[0].isColourFilter
	numberOfDimensions = RFfiltersPropertiesList2[0].numberOfDimensions
	RFtype = RFfiltersPropertiesList2[0].RFtype
	filterApplicationResultThreshold = ATORpt_RFfilter.calculateFilterApplicationResultThreshold(filterApplicationResult, ATORpt_RFfilter.minimumFilterRequirement, filterSize, isColourFilter, numberOfDimensions, RFtype)
	filterApplicationResultThresholdIndices = pt.nonzero(filterApplicationResultThreshold).squeeze()

	if not ATORpt_RFoperations.isTensorEmpty(filterApplicationResultThresholdIndices):
		filterApplicationResultThresholded = filterApplicationResult[filterApplicationResultThresholdIndices]
		
		filterApplicationResultThresholdIndicesList = filterApplicationResultThresholdIndices.tolist()
		filterApplicationResultThresholdedList = filterApplicationResultThresholded.tolist()

		for RFthresholdedListIndex, RFlistIndex in enumerate(filterApplicationResultThresholdIndicesList):
			imageSegmentIndex, RFfilterIndex = divmod(RFlistIndex, len(RFfiltersTensor))
			centerCoordinates1, centerCoordinates2 = divmod(imageSegmentIndex, resolutionProperties.imageSize[1])
			centerCoordinates = (centerCoordinates1, centerCoordinates2)

			if(not RFuseParallelProcessedCNN):
				RFImage = inputImageSegments[imageSegmentIndex]
			RFfiltersProperties = RFfiltersPropertiesList2[RFfilterIndex]
			RFproperties = copy.deepcopy(RFfiltersProperties)
			RFproperties.centerCoordinates = centerCoordinates
			RFproperties.filterIndex = RFfilterIndex
			RFproperties.filterApplicationResult = filterApplicationResultThresholdedList[RFthresholdedListIndex]
			RFproperties.imageSegmentIndex = imageSegmentIndex
			RFpropertiesList.append(RFproperties)

	return RFpropertiesList

def generateRFtypeTriFromPointFeatureSets(resolutionProperties, RFpropertiesListPointFeatures):
	RFpropertiesList = []
	if(len(RFpropertiesListPointFeatures) >= triNumberPointFeatures):
		#generate RFtypeTri for sets of features
		coordinatesList = []
		for RFpropertiesPointFeature in RFpropertiesListPointFeatures:
			coordinatesList.append(RFpropertiesPointFeature.centerCoordinates)
		coordinates = np.array(coordinatesList)
		candidates = np.array(coordinatesList)
		closestCoordinates, closestIndices = findKclosestCoordinates2D(coordinates, candidates, triNumberPointFeatures)
		for RFpropertiesPointFeatureIndex, RFpropertiesPointFeature in enumerate(RFpropertiesListPointFeatures):
			print("deriveRFtriPropertiesFromPointFeatureSet: closestIndices[RFpropertiesPointFeatureIndex] = ", closestIndices[RFpropertiesPointFeatureIndex])
			RFpropertiesTri = deriveRFtriPropertiesFromPointFeatureSet(resolutionProperties, RFpropertiesListPointFeatures, closestIndices[RFpropertiesPointFeatureIndex])
			if(uniqueRFCandidate(RFpropertiesList, RFproperties)):
				RFpropertiesList.append(RFpropertiesTri)
	return RFpropertiesList
	
def findKclosestCoordinates2D(coordinates, candidates, k):
	print("coordinates = ", coordinates)
	print("candidates = ", candidates)
	candidates = candidates[:, np.newaxis, :]
	distances = np.sqrt(np.sum((coordinates - candidates)**2, axis=2))
	closestIndices = np.argsort(distances, axis=1)[:, :k]
	closestCoordinates = np.take_along_axis(coordinates, closestIndices, axis=0)
	return closestCoordinates, closestIndices

def findKclosestCoordinates3D(coordinates, candidates, k):
	candidates = candidates[:, np.newaxis, :]
	distances = np.sqrt(np.sum((coordinates - candidates)**2, axis=2))
	closestIndices = np.argsort(distances, axis=1)[:, :k]
	closestCoordinates = np.take_along_axis(coordinates, closestIndices, axis=0)
	return closestCoordinates, closestIndices

def deriveRFtriPropertiesFromPointFeatureSet(resolutionProperties, RFpropertiesListPointFeatures, RFpropertiesPointFeaturesTriIndices):
	filterColour = RFpropertiesListPointFeatures[0].colour
	filterCenterCoordinates, filterSize, axesLength, angle = calculateTriangleDimensions(RFpropertiesPointFeaturesTriIndices)
	RFpropertiesTri = ATORpt_RFproperties.RFpropertiesClass(resolutionProperties.resolutionIndex, resolutionProperties.resolutionFactor, filterSize, RFtypeTri, filterCenterCoordinates, axesLength, angle, filterColour)
	RFpropertiesTri.isColourFilter = RFpropertiesListPointFeatures[0].isColourFilter
	return RFpropertiesTri

def uniqueRFCandidate(RFpropertiesList, RFpropertiesCandidate):
	#requires optimisation
	uniqueCandidate = True
	for RFproperties in RFpropertiesList:
		if(RFpropertiesCandidate.centerCoordinates == RFproperties.centerCoordinates):	#CHECKTHIS (assumes no concentric feature triangles in image)
			uniqueCandidate = False
	return uniqueCandidate

def distance(x1, y1, x2, y2):
	return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculateTriangleDimensions(coordinates):
	centerCoordinates = np.mean(coordinates)
	x1, y1 = coordinates[0]
	x2, y2 = coordinates[1]
	x3, y3 = coordinates[2]
	filterSize = ((max(x1, x2, x3) - min(x1, x2, x3)), (max(y1, y2, y3) - min(y1, y2, y3)))
	base_midpoint_x = (x1 + x2) / 2
	base_midpoint_y = (y1 + y2) / 2
	height = distance(base_midpoint_x, base_midpoint_y, x3, y3)
	width = distance(x1, y1, x2, y2)
	angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
	axesLength = (width, height)	#CHECKTHIS
	return centerCoordinates, filterSize, axesLength, angle
	
	
