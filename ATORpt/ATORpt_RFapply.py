"""ATORpt_RFapply.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORpt_RFmainFT.py

# Usage:
See ATORpt_RFmainFT.py

# Description:
ATORpt RF apply

"""

import torch as pt
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import copy
import math

from ATORpt_RFglobalDefs import *
import ATORpt_RFellipsePropertiesClass
import ATORpt_RFgenerateEllipse
import ATORpt_RFgenerateTri
import ATORpt_RFapplyFilter
import ATORpt_RFpropertiesClass
import ATORpt_RFapplyFilter
import ATORpt_RFoperations
import ATORpt_operations
if(RFuseParallelProcessedCNN):
	import ATORpt_RFapplyCNN

class RFneuronClass():
	def __init__(self, resolutionProperties, RFproperties, RFfilter, RFImage):
		self.resolutionIndex = resolutionProperties.resolutionIndex
		self.RFproperties = RFproperties
		# if(RFsaveRFfiltersAndImageSegments):
		self.RFfilter = RFfilter
		self.RFneuronParents = []
		self.RFneuronComponents = []

class ATORneuronClass():
	def __init__(self, resolutionProperties, RFproperties, RFfilter, RFImage):
		self.resolutionIndex = resolutionProperties.resolutionIndex
		self.RFproperties = RFproperties
		self.RFpropertiesNormalised = normaliseLocalRFproperties(RFproperties)
		self.RFpropertiesNormalisedWRTparent = None
		self.RFpropertiesNormalisedGlobal = normaliseGlobalRFproperties(RFproperties, resolutionProperties.resolutionFactor)
		if RFsaveRFfiltersAndImageSegments:
			if(RFsaveRFfilters):
				self.RFfilter = RFfilter
				self.RFfilterNormalised = ATORpt_RFapplyFilter.normaliseRFfilter(RFfilter, RFproperties)
				self.RFfilterNormalisedWRTparent = None
			if(RFsaveRFimageSegments):
				self.RFImage = RFImage
				self.RFImageNormalised = ATORpt_RFapplyFilter.normaliseRFfilter(RFImage, RFproperties)
				self.RFImageNormalisedWRTparent = None
		self.neuronComponents = []
		self.neuronComponentsWeightsList = []

def updateRFhierarchyAccelerated(RFfiltersListAllRes, RFfiltersPropertiesListAllRes, ATORneuronListAllLayers, inputimagefilename):
	
	print("imageSizeBase = ", imageSizeBase)
	inputImage = cv2.imread(inputimagefilename)

	inputImage = cv2.resize(inputImage, imageSizeBase)
	inputImageRGB = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
	inputImageGray = cv2.cvtColor(inputImageRGB, cv2.COLOR_RGB2GRAY)

	inputImageRGBTensor = pt.tensor(inputImageRGB, dtype=pt.float32).permute(2, 0, 1).unsqueeze(0)
	inputImageGrayTensor = pt.tensor(inputImageGray, dtype=pt.float32).unsqueeze(0).unsqueeze(0)

	inputImageHeight, inputImageWidth, inputImageChannels = inputImage.shape
	print("inputImageHeight = ", inputImageHeight, "inputImageWidth = ", inputImageWidth, ", inputImageChannels = ", inputImageChannels)
	blankArray = np.full((inputImageHeight, inputImageWidth, 3), 255, np.uint8)
	outputImage = blankArray
	
	inputImageRGBSegmentsAllRes = []
	inputImageGraySegmentsAllRes = []

	if debugLowIterations:
		resolutionIndexMax = 1
	else:
		resolutionIndexMax = numberOfResolutions

	if(RFuseParallelProcessedCNN):
		for resolutionIndex in range(resolutionIndexFirst, resolutionIndexMax):
			resolutionProperties = ATORpt_RFoperations.RFresolutionProperties(resolutionIndex, resolutionIndexFirst, numberOfResolutions, imageSizeBase)
			RFfiltersFeatureTypeConvLayerList = ATORpt_RFapplyCNN.generateCNNfilters(RFfiltersListAllRes[resolutionIndex], RFfiltersPropertiesListAllRes[resolutionIndex], resolutionProperties)
			inputImageRGBTensorResized = F.interpolate(inputImageRGBTensor, size=resolutionProperties.imageSize, mode='bilinear', align_corners=False)
			inputImageGrayTensorResized = F.interpolate(inputImageGrayTensor, size=resolutionProperties.imageSize, mode='bilinear', align_corners=False)
			#inputImageGrayTensorResized.unsqueeze(1)	#add rgb channels
			#inputImageGrayTensorResized = inputImageGrayTensorResized.repeat(1, rgbNumChannels, 1, 1)
			RFfiltersFeatureTypeList = RFfiltersListAllRes[resolutionIndex]
			RFfiltersPropertiesFeatureTypeList = RFfiltersPropertiesListAllRes[resolutionIndex]
			applyRFfiltersList(resolutionProperties, RFfiltersFeatureTypeList, RFfiltersPropertiesFeatureTypeList, ATORneuronListAllLayers, inputImageRGBTensorResized=inputImageRGBTensorResized, inputImageGrayTensorResized=inputImageGrayTensorResized, RFfiltersFeatureTypeConvLayerList=RFfiltersFeatureTypeConvLayerList)
	else:
		for resolutionIndex in range(resolutionIndexFirst, resolutionIndexMax):
			resolutionProperties = ATORpt_RFoperations.RFresolutionProperties(resolutionIndex, resolutionIndexFirst, numberOfResolutions, imageSizeBase)
			inputImageRGBTensorResized = F.interpolate(inputImageRGBTensor, size=resolutionProperties.imageSize, mode='bilinear', align_corners=False)
			inputImageGrayTensorResized = F.interpolate(inputImageGrayTensor, size=resolutionProperties.imageSize, mode='bilinear', align_corners=False)
			inputImageRGBSegments, inputImageGraySegments = generateImageSegments(resolutionProperties, inputImageRGBTensorResized, inputImageGrayTensorResized)
			inputImageRGBSegmentsAllRes.append(inputImageRGBSegments)
			inputImageGraySegmentsAllRes.append(inputImageGraySegments)
		for resolutionIndex in range(resolutionIndexFirst, resolutionIndexMax):
			resolutionProperties = ATORpt_RFoperations.RFresolutionProperties(resolutionIndex, resolutionIndexFirst, numberOfResolutions, imageSizeBase)
			inputImageRGBSegments = inputImageRGBSegmentsAllRes[resolutionIndex]
			inputImageGraySegments = inputImageGraySegmentsAllRes[resolutionIndex]
			RFfiltersFeatureTypeList = RFfiltersListAllRes[resolutionIndex]
			RFfiltersPropertiesFeatureTypeList = RFfiltersPropertiesListAllRes[resolutionIndex]
			applyRFfiltersList(resolutionProperties, RFfiltersFeatureTypeList, RFfiltersPropertiesFeatureTypeList, ATORneuronListAllLayers, inputImageRGBSegments=inputImageRGBSegments, inputImageGraySegments=inputImageGraySegments)
			
if(RFuseParallelProcessedCNN):
	def generateImageSegment(resolutionProperties, RFproperties, inputImageTensor, isColourFilter):
		filterRadius = RFproperties.imageSize[0]
		centerCoordinates = (RFproperties.imageSegmentIndex//resolutionProperties.imageSize[1], RFproperties.imageSegmentIndex%resolutionProperties.imageSize[1])
		inputImageSegment = generateImageSegmentType(resolutionProperties, centerCoordinates, filterRadius, inputImageTensor)
		inputImageSegment = inputImageSegment[0]
		inputImageSegment = inputImageSegment.permute(1, 2, 0)	#ensure channels dim is last #match RFfilter dims: height, width, channels	#see transformRFfilterTF2D
		return inputImageSegment
else:
	def generateImageSegments(resolutionProperties, inputImageRGBTensor, inputImageGrayTensor):
		inputImageRGBSegmentsList = []
		inputImageGraySegmentsList = []

		axesLengthMax, filterRadius, filterSize = ATORpt_RFapplyFilter.getFilterDimensions(resolutionProperties)

		if debugVerbose:
			print("")
			print("resolutionIndex = ", resolutionProperties.resolutionIndex)
			print("resolutionFactor = ", resolutionProperties.resolutionFactor)
			print("imageSize = ", resolutionProperties.imageSize)
			print("filterRadius = ", filterRadius)
			print("axesLengthMax = ", axesLengthMax)
			print("filterSize = ", filterSize)

		imageSegmentIndex = 0
		for centerCoordinates1 in range(0, resolutionProperties.imageSize[0], ellipseCenterCoordinatesResolution):
			for centerCoordinates2 in range(0, resolutionProperties.imageSize[1], ellipseCenterCoordinatesResolution):
				centerCoordinates = (centerCoordinates1, centerCoordinates2)
				inputImageRGBSegment = generateImageSegmentType(resolutionProperties, centerCoordinates, filterRadius, inputImageRGBTensor)
				inputImageGraySegment = generateImageSegmentType(resolutionProperties, centerCoordinates, filterRadius, inputImageGrayTensor)
				inputImageRGBSegmentsList.append(inputImageRGBSegment)
				inputImageGraySegmentsList.append(inputImageGraySegment)
				imageSegmentIndex = imageSegmentIndex + 1

		inputImageRGBSegments = pt.cat(inputImageRGBSegmentsList, dim=0)
		inputImageGraySegments = pt.cat(inputImageGraySegmentsList, dim=0)

		return inputImageRGBSegments, inputImageGraySegments

def generateImageSegmentType(resolutionProperties, centerCoordinates, filterRadius, inputImageTensor):
	allFilterCoordinatesWithinImageResult, imageSegmentStart, imageSegmentEnd = ATORpt_RFapplyFilter.allFilterCoordinatesWithinImage(centerCoordinates, filterRadius, resolutionProperties.imageSize)
	inputImageSegment = inputImageTensor[:, :, imageSegmentStart[0]:imageSegmentEnd[0], imageSegmentStart[1]:imageSegmentEnd[1]]
	if(ATORpt_RFoperations.storeRFfiltersValuesAsFractions):
		inputImageSegment = pt.divide(inputImageSegment, ATORpt_RFoperations.rgbMaxValue)	#CHECKTHIS is same for isColourFilter
	return inputImageSegment
			
			
def applyRFfiltersList(resolutionProperties, RFfiltersFeatureTypeList, RFfiltersPropertiesFeatureTypeList, ATORneuronListAllLayers, inputImageRGBSegments=None, inputImageGraySegments=None, inputImageRGBTensorResized=None, inputImageGrayTensorResized=None, RFfiltersFeatureTypeConvLayerList=None):
	print("\tapplyRFfiltersList: resolutionIndex = ", resolutionProperties.resolutionIndex)

	ATORneuronListArray = ATORneuronListAllLayers[resolutionProperties.resolutionIndex]

	for RFfeatureTypeListIndex in range(len(RFfiltersPropertiesFeatureTypeList)):
		RFfiltersList = RFfiltersFeatureTypeList[RFfeatureTypeListIndex]
		RFfiltersPropertiesList = RFfiltersPropertiesFeatureTypeList[RFfeatureTypeListIndex]
		if(RFuseParallelProcessedCNN):
			if(RFuseParallelProcessedCNNRFchannelsImplementation==1):
				RFfiltersConvLayer = RFfiltersFeatureTypeConvLayerList[RFfeatureTypeListIndex]
			elif(RFuseParallelProcessedCNNRFchannelsImplementation > 1):
				RFfiltersConvLayerList = RFfiltersFeatureTypeConvLayerList[RFfeatureTypeListIndex]
		#axesLengthMax, filterRadius, filterSize = ATORpt_RFapplyFilter.getFilterDimensions(resolutionProperties)	#OLD
		if(RFfeatureTypeListIndex == RFfeatureTypeIndexEllipse):
			axesLengthMax, filterRadius, filterSize = ATORpt_RFgenerateEllipse.getFilterDimensions(resolutionProperties)
		elif(RFfeatureTypeListIndex == RFfeatureTypeIndexTri):
			axesLengthMax, filterRadius, filterSize = ATORpt_RFgenerateTri.getFilterDimensions(resolutionProperties)
		
		for RFlistIndex1 in range(len(RFfiltersPropertiesList)):
			print("RFlistIndex1 = ", RFlistIndex1)
			RFfiltersTensor = RFfiltersList[RFlistIndex1]
			RFfiltersPropertiesList2 = RFfiltersPropertiesList[RFlistIndex1]
			if(RFuseParallelProcessedCNNRFchannelsImplementation > 1):
				RFfiltersConv = RFfiltersConvLayerList[RFlistIndex1]
			isColourFilter = RFfiltersPropertiesList2[0].isColourFilter
			numberOfDimensions = RFfiltersPropertiesList2[0].numberOfDimensions
			
			if(RFuseParallelProcessedCNN):
				if isColourFilter:
					inputImage = inputImageRGBTensorResized
				else:
					inputImage = inputImageGrayTensorResized
				RFpropertiesList = applyRFfilters(resolutionProperties, RFfiltersTensor, numberOfDimensions, RFfiltersPropertiesList2, inputImage=inputImage, RFfiltersConv=RFfiltersConv)
			else:
				if isColourFilter:
					inputImageSegments = inputImageRGBSegments
				else:
					inputImageSegments = inputImageGraySegments
				RFpropertiesList = applyRFfilters(resolutionProperties, RFfiltersTensor, numberOfDimensions, RFfiltersPropertiesList2, inputImageSegments=inputImageSegments)

			if(RFdetectTriFeaturesSeparately):
				if(RFfiltersPropertiesList2[0].RFtype == RFtypeTemporaryPointFeatureKernel):
					#print("RFpropertiesList = ", RFpropertiesList)
					RFpropertiesList = generateRFtypeTriFromPointFeatureSets(resolutionProperties, RFpropertiesList)	#generate RFtypeTri for sets of features
					#FUTURE: consider recreating RFfiltersTensor also (not currently required)
					
			#print("ATORneuronList append: len(filterApplicationResultThresholdIndicesList) = ", len(filterApplicationResultThresholdIndicesList))
			for RFproperties in RFpropertiesList:

				filterApplicationResult = RFproperties.filterApplicationResult
				
				if(RFsaveRFimageSegments):
					if(RFuseParallelProcessedCNN):
						if isColourFilter:
							inputImageTensor = inputImageRGBTensorResized
						else:
							inputImageTensor = inputImageGrayTensorResized
						imageSegment = generateImageSegment(resolutionProperties, RFproperties, inputImageTensor, isColourFilter)
					else:
						imageSegment = inputImageSegments[RFproperties.imageSegmentIndex]
				
				RFfilter = None
				RFImage = None
				if RFsaveRFfiltersAndImageSegments:
					RFfilter = RFfiltersTensor[RFproperties.filterIndex]
					if(RFsaveRFimageSegments):
						RFImage = imageSegment

				centerCoordinates = RFproperties.centerCoordinates
				allFilterCoordinatesWithinImageResult, imageSegmentStart, imageSegmentEnd = ATORpt_RFapplyFilter.allFilterCoordinatesWithinImage(centerCoordinates, filterRadius, resolutionProperties.imageSize)
				if allFilterCoordinatesWithinImageResult:

					neuron = ATORneuronClass(resolutionProperties, RFproperties, RFfilter, RFImage)

					ATORneuronList = ATORneuronListArray[centerCoordinates[0]][centerCoordinates[1]]
					if ATORneuronList is None:
						ATORneuronList = []
						ATORneuronListArray[centerCoordinates[0]][centerCoordinates[1]] = ATORneuronList
					ATORneuronList.append(neuron)

					foundParentNeuron, parentNeuron = findParentNeuron(ATORneuronListAllLayers, resolutionProperties.resolutionIndex, resolutionProperties, neuron)
					if foundParentNeuron:
						print("foundParentNeuron")
						parentNeuron.neuronComponents.append(neuron)
						normaliseRFComponentWRTparent(resolutionProperties, neuron, parentNeuron.RFproperties)
						parentNeuron.neuronComponentsWeightsList.append(filterApplicationResult)

def findParentNeuron(ATORneuronListAllLayers, resolutionIndexLast, resolutionPropertiesChild, neuronChild):
	foundParentNeuron = False
	parentNeuron = None
	if resolutionIndexLast > resolutionPropertiesChild.resolutionIndexFirst:
		resolutionIndex = resolutionIndexLast - 1
		ATORneuronListArray = ATORneuronListAllLayers[resolutionIndex]

		resolutionProperties = copy.deepcopy(resolutionPropertiesChild)
		resolutionProperties.resolutionIndex = resolutionIndex
		resolutionProperties.resolutionFactor, resolutionProperties.resolutionFactorReverse, resolutionProperties.imageSize = ATORpt_RFoperations.getImageDimensionsR(resolutionProperties)

		axesLengthMax, filterRadius, filterSize = ATORpt_RFapplyFilter.getFilterDimensions(resolutionProperties)

		RFrangeGlobalMin, RFrangeGlobalMax = getRFrangeAtResolutionGlobal(resolutionPropertiesChild, neuronChild.RFproperties.centerCoordinates)
		RFrangeLocalMin, RFrangeLocalMax = getRFrangeAtResolutionLocal(resolutionProperties, RFrangeGlobalMin, RFrangeGlobalMax)
		for centerCoordinates1 in range(RFrangeLocalMin[0], RFrangeLocalMax[0] + 1, ellipseCenterCoordinatesResolution):
			for centerCoordinates2 in range(RFrangeLocalMin[1], RFrangeLocalMax[1] + 1, ellipseCenterCoordinatesResolution):
				centerCoordinates = [centerCoordinates1, centerCoordinates2]

				allFilterCoordinatesWithinImageResult, imageSegmentStart, imageSegmentEnd = ATORpt_RFapplyFilter.allFilterCoordinatesWithinImage(centerCoordinates, filterRadius, resolutionProperties.imageSize)
				if allFilterCoordinatesWithinImageResult:

					ATORneuronList = ATORneuronListArray[centerCoordinates[0]][centerCoordinates[1]]
					if ATORneuronList is not None:
						for neuron in ATORneuronList:
							if childRFoverlapsParentRF(neuronChild.RFpropertiesNormalisedGlobal, neuron.RFpropertiesNormalisedGlobal):
								foundParentNeuron = True
								parentNeuron = neuron

		if not foundParentNeuron:
			foundParentNeuron, neuronFound = findParentNeuron(ATORneuronListAllLayers, resolutionIndex, resolutionPropertiesChild, neuronChild)

	return foundParentNeuron, parentNeuron

def getRFrangeAtResolutionGlobal(resolutionProperties, centerCoordinates):
	axesLengthMax, filterRadius, filterSize = ATORpt_RFapplyFilter.getFilterDimensions(resolutionProperties)

	RFrangeGlobalCentre = [int(centerCoordinates[0] * resolutionProperties.resolutionFactor), int(centerCoordinates[1] * resolutionProperties.resolutionFactor)]
	RFrangeGlobalSize = [int(filterSize[0] * resolutionProperties.resolutionFactor), int(filterSize[1] * resolutionProperties.resolutionFactor)]
	RFrangeGlobalMin = [RFrangeGlobalCentre[0] - RFrangeGlobalSize[0], RFrangeGlobalCentre[1] - RFrangeGlobalSize[1]]
	RFrangeGlobalMax = [RFrangeGlobalCentre[0] + RFrangeGlobalSize[0], RFrangeGlobalCentre[1] + RFrangeGlobalSize[1]]
	return RFrangeGlobalMin, RFrangeGlobalMax

def getRFrangeAtResolutionLocal(resolutionProperties, RFrangeGlobalMin, RFrangeGlobalMax):
	RFrangeLocalMin = [int(RFrangeGlobalMin[0] / resolutionProperties.resolutionFactor), int(RFrangeGlobalMin[1] / resolutionProperties.resolutionFactor)]
	RFrangeLocalMax = [int(RFrangeGlobalMax[0] / resolutionProperties.resolutionFactor), int(RFrangeGlobalMax[1] / resolutionProperties.resolutionFactor)]
	return RFrangeLocalMin, RFrangeLocalMax
	

def normaliseRFComponentWRTparent(resolutionProperties, neuronComponent, RFpropertiesParent):
	neuronComponent.RFpropertiesNormalisedWRTparent = ATORpt_RFpropertiesClass.generateRFtransformedProperties(neuronComponent, RFpropertiesParent)
	if RFsaveRFfiltersAndImageSegments:
		neuronComponent.RFfilterNormalisedWRTparent = ATORpt_RFapplyFilter.transformRFfilterTF(neuronComponent.RFfilter, RFpropertiesParent)
		if(RFsaveRFimageSegments):
			neuronComponent.RFImageNormalisedWRTparent = ATORpt_RFapplyFilter.transformRFfilterTF(neuronComponent.RFImage, RFpropertiesParent)


def childRFoverlapsParentRF(neuronRFpropertiesNormalisedGlobal, lowerNeuronRFpropertiesNormalisedGlobal):
	if neuronRFpropertiesNormalisedGlobal.RFtype == RFtypeEllipse:
		return ATORpt_RFellipsePropertiesClass.centroidOverlapsEllipse(neuronRFpropertiesNormalisedGlobal, lowerNeuronRFpropertiesNormalisedGlobal)
	elif neuronRFpropertiesNormalisedGlobal.RFtype == RFtypeTri:
		# CHECKTHIS is appropriate for tri;
		return ATORpt_RFellipsePropertiesClass.centroidOverlapsEllipse(neuronRFpropertiesNormalisedGlobal, lowerNeuronRFpropertiesNormalisedGlobal)


def normaliseGlobalRFproperties(RFproperties, resolutionFactor):
	# normalise RF respect to original image size
	if RFproperties.RFtype == RFtypeEllipse:
		RFpropertiesNormalisedGlobal = ATORpt_RFgenerateEllipse.normaliseGlobalEllipseProperties(RFproperties, resolutionFactor)
	elif RFproperties.RFtype == RFtypeTri:
		RFpropertiesNormalisedGlobal = ATORpt_RFgenerateTri.normaliseGlobalTriProperties(RFproperties, resolutionFactor)
	return RFpropertiesNormalisedGlobal


def normaliseLocalRFproperties(RFproperties):
	# normalise ellipse respect to major/minor ellipticity axis orientation (WRT self)
	if RFproperties.RFtype == RFtypeEllipse:
		ATORpt_RFgenerateEllipse.normaliseLocalEllipseProperties(RFproperties)
	elif RFproperties.RFtype == RFtypeTri:
		# CHECKTHIS is appropriate for tri; (ellipseNormalisedAxesLength, ellipseNormalisedAxesLength/q) might create equilateral triangle
		ATORpt_RFgenerateTri.normaliseLocalTriProperties(RFproperties)


def applyRFfilters(resolutionProperties, RFfiltersTensor, numberOfDimensions, RFfiltersPropertiesList2, inputImageSegments=None, inputImage=None, RFfiltersConv=None):
	# perform convolution for each filter size;
	axesLengthMax, filterRadius, filterSize = ATORpt_RFapplyFilter.getFilterDimensions(resolutionProperties)
	RFpropertiesList = []
	
	if(RFuseParallelProcessedCNN):
		isColourFilter = RFfiltersPropertiesList2[0].isColourFilter
		numberOfKernels = RFfiltersPropertiesList2[0].numberOfKernels
		#print("isColourFilter = ", isColourFilter)
		#print("numberOfKernels = ", numberOfKernels)
		filterApplicationResult = ATORpt_RFapplyCNN.applyCNNfilters(inputImage, RFfiltersConv, isColourFilter, numberOfKernels)	#dim: numberOfImageSegments*numberOfKernels
	else:
		inputImageSegmentsPixelsFlattened = inputImageSegments.view(inputImageSegments.shape[0], -1)
		RFfiltersTensorPixelsFlattened = RFfiltersTensor.view(RFfiltersTensor.shape[0], -1)
		filterApplicationResult = pt.matmul(inputImageSegmentsPixelsFlattened, RFfiltersTensorPixelsFlattened.t())
		filterApplicationResult = filterApplicationResult.view(-1)	#dim: numberOfImageSegments*numberOfKernels

	isColourFilter = RFfiltersPropertiesList2[0].isColourFilter
	numberOfDimensions = RFfiltersPropertiesList2[0].numberOfDimensions
	RFtype = RFfiltersPropertiesList2[0].RFtype
	filterApplicationResultThreshold = ATORpt_RFapplyFilter.calculateFilterApplicationResultThreshold(filterApplicationResult, ATORpt_RFapplyFilter.minimumFilterRequirement, filterSize, isColourFilter, numberOfDimensions, RFtype)
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
		closestCoordinates, closestIndices = ATORpt_operations.findKclosestCoordinates2D(coordinates, candidates, triNumberPointFeatures)
		for RFpropertiesPointFeatureIndex, RFpropertiesPointFeature in enumerate(RFpropertiesListPointFeatures):
			print("deriveRFtriPropertiesFromPointFeatureSet: closestIndices[RFpropertiesPointFeatureIndex] = ", closestIndices[RFpropertiesPointFeatureIndex])
			RFpropertiesTri = deriveRFtriPropertiesFromPointFeatureSet(resolutionProperties, RFpropertiesListPointFeatures, closestIndices[RFpropertiesPointFeatureIndex])
			if(uniqueRFCandidate(RFpropertiesList, RFproperties)):
				RFpropertiesList.append(RFpropertiesTri)
	return RFpropertiesList
	
def deriveRFtriPropertiesFromPointFeatureSet(resolutionProperties, RFpropertiesListPointFeatures, RFpropertiesPointFeaturesTriIndices):
	filterColour = RFpropertiesListPointFeatures[0].colour
	filterCenterCoordinates, filterSize, axesLength, angle = calculateTriangleDimensions(RFpropertiesPointFeaturesTriIndices)
	RFpropertiesTri = ATORpt_RFpropertiesClass.RFpropertiesClass(resolutionProperties.resolutionIndex, resolutionProperties.resolutionFactor, filterSize, RFtypeTri, filterCenterCoordinates, axesLength, angle, filterColour)
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
	
	
