"""ATORpt_RFmain.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORpt_main.py

# Usage:
source activate pytorchsenv
python ATORpt_RFmain.py images/leaf1.png

# Description:
ATORpt is a hardware accelerated version of BAI ATOR (Axis Transformation Object Recognition) for TensorFlow.

ATORpt supports ellipsoid features, and normalises them with respect to their major/minor ellipticity axis orientation. 

There are a number of advantages of using ellipsoid features over point features;
* the number of feature sets/normalised snapshots required is significantly reduced
* scene component structure can be maintained (as detected component ellipses can be represented in a hierarchical graph structure)

Ellipse features/components are detected based on simulated artificial receptive fields; RF (on/off, off/on).

ATORpt also supports point (corner/centroid) features of the ATOR specification; 
https://www.wipo.int/patentscope/search/en/WO2011088497

# Future:
Requires upgrading to support 3DOD receptive field detection (ellipses/ellipsoids/features in 3D space)

"""

import os
import torch as pt
import numpy as np
import click
import cv2
import copy
import sys
import torch
import torch.nn.functional as F

import ATORpt_RFdetectEllipses
import ATORpt_RFgenerateApply
import ATORpt_RFproperties
import ATORpt_RFfilter
import ATORpt_RFoperations

np.set_printoptions(threshold=sys.maxsize)

generateRFfiltersEllipse = True
generateRFfiltersTri = False

debugLowIterations = False
debugVerbose = True
debugSaveRFfiltersAndImageSegments = True

resolutionIndexFirst = 0
numberOfResolutions = 4

ellipseCenterCoordinatesResolution = 1
imageSizeBase = (256, 256)

class ATORneuronClass():
	def __init__(self, resolutionProperties, RFproperties, RFfilter, RFImage):
		self.resolutionIndex = resolutionProperties.resolutionIndex
		self.RFproperties = RFproperties
		self.RFpropertiesNormalised = ATORpt_RFgenerateApply.normaliseLocalRFproperties(RFproperties)
		self.RFpropertiesNormalisedWRTparent = None
		self.RFpropertiesNormalisedGlobal = ATORpt_RFgenerateApply.normaliseGlobalRFproperties(RFproperties, resolutionProperties.resolutionFactor)
		if debugSaveRFfiltersAndImageSegments:
			self.RFfilter = RFfilter
			self.RFfilterNormalised = ATORpt_RFfilter.normaliseRFfilter(RFfilter, RFproperties)
			self.RFfilterNormalisedWRTparent = None
			self.RFImage = RFImage
			self.RFImageNormalised = ATORpt_RFfilter.normaliseRFfilter(RFImage, RFproperties)
			self.RFImageNormalisedWRTparent = None
		self.neuronComponents = []
		self.neuronComponentsWeightsList = []

def prepareRFhierarchyAccelerated():
	RFfiltersListAllRes = []
	RFfiltersPropertiesListAllRes = []
	ATORneuronListAllLayers = []

	if debugLowIterations:
		resolutionIndexMax = 1
	else:
		resolutionIndexMax = numberOfResolutions

	for resolutionIndex in range(resolutionIndexFirst, resolutionIndexMax):
		resolutionProperties = ATORpt_RFoperations.RFresolutionProperties(resolutionIndex, resolutionIndexFirst, numberOfResolutions, imageSizeBase, debugVerbose, debugSaveRFfiltersAndImageSegments)
		RFfiltersList, RFfiltersPropertiesList = ATORpt_RFgenerateApply.generateRFfilters(resolutionProperties, generateRFfiltersEllipse, generateRFfiltersTri)
		RFfiltersListAllRes.append(RFfiltersList)
		RFfiltersPropertiesListAllRes.append(RFfiltersPropertiesList)
	
	for resolutionIndex in range(resolutionIndexFirst, resolutionIndexMax):
		resolutionProperties = ATORpt_RFoperations.RFresolutionProperties(resolutionIndex, resolutionIndexFirst, numberOfResolutions, imageSizeBase, debugVerbose, debugSaveRFfiltersAndImageSegments)
		ATORneuronListArray = initialiseATORneuronListArray(resolutionProperties)
		ATORneuronListAllLayers.append(ATORneuronListArray)
	
	return RFfiltersListAllRes, RFfiltersPropertiesListAllRes, ATORneuronListAllLayers

def updateRFhierarchyAccelerated(RFfiltersListAllRes, RFfiltersPropertiesListAllRes, ATORneuronListAllLayers, inputimagefilename):
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

	for resolutionIndex in range(resolutionIndexFirst, resolutionIndexMax):
		resolutionProperties = ATORpt_RFoperations.RFresolutionProperties(resolutionIndex, resolutionIndexFirst, numberOfResolutions, imageSizeBase, debugVerbose, debugSaveRFfiltersAndImageSegments)
		inputImageRGBTensorResized = F.interpolate(inputImageRGBTensor, size=resolutionProperties.imageSize, mode='bilinear', align_corners=False)
		inputImageGrayTensorResized = F.interpolate(inputImageGrayTensor, size=resolutionProperties.imageSize, mode='bilinear', align_corners=False)
		inputImageRGBSegments, inputImageGraySegments = generateImageSegments(resolutionProperties, inputImageRGBTensorResized, inputImageGrayTensorResized)
		inputImageRGBSegmentsAllRes.append(inputImageRGBSegments)
		inputImageGraySegmentsAllRes.append(inputImageGraySegments)

	for resolutionIndex in range(resolutionIndexFirst, resolutionIndexMax):
		resolutionProperties = ATORpt_RFoperations.RFresolutionProperties(resolutionIndex, resolutionIndexFirst, numberOfResolutions, imageSizeBase, debugVerbose, debugSaveRFfiltersAndImageSegments)
		inputImageRGBSegments = inputImageRGBSegmentsAllRes[resolutionIndex]
		inputImageGraySegments = inputImageGraySegmentsAllRes[resolutionIndex]
		RFfiltersList = RFfiltersListAllRes[resolutionIndex]
		RFfiltersPropertiesList = RFfiltersPropertiesListAllRes[resolutionIndex]
		applyRFfiltersList(resolutionProperties, inputImageRGBSegments, inputImageGraySegments, RFfiltersList, RFfiltersPropertiesList, ATORneuronListAllLayers)

def initialiseATORneuronListArray(resolutionProperties):
	size = (resolutionProperties.imageSize[0], resolutionProperties.imageSize[1])
	ATORneuronListArray = [[None for _ in range(size[1])] for _ in range(size[0])]
	return ATORneuronListArray

def generateImageSegments(resolutionProperties, inputImageRGBTensor, inputImageGrayTensor):
	inputImageRGBSegmentsList = []
	inputImageGraySegmentsList = []

	axesLengthMax, filterRadius, filterSize = ATORpt_RFfilter.getFilterDimensions(resolutionProperties)

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
			allFilterCoordinatesWithinImageResult, imageSegmentStart, imageSegmentEnd = ATORpt_RFfilter.allFilterCoordinatesWithinImage(centerCoordinates, filterRadius, resolutionProperties.imageSize)
			inputImageRGBSegment = inputImageRGBTensor[:, :, imageSegmentStart[0]:imageSegmentEnd[0], imageSegmentStart[1]:imageSegmentEnd[1]]
			inputImageGraySegment = inputImageGrayTensor[:, :, imageSegmentStart[0]:imageSegmentEnd[0], imageSegmentStart[1]:imageSegmentEnd[1]]
			if(ATORpt_RFoperations.storeRFfiltersValuesAsFractions):
				inputImageRGBSegment = pt.divide(inputImageRGBSegment, ATORpt_RFoperations.rgbMaxValue)
				inputImageGraySegment = pt.divide(inputImageGraySegment, ATORpt_RFoperations.rgbMaxValue)
			inputImageRGBSegmentsList.append(inputImageRGBSegment)
			inputImageGraySegmentsList.append(inputImageGraySegment)
			imageSegmentIndex = imageSegmentIndex + 1

	inputImageRGBSegments = pt.cat(inputImageRGBSegmentsList, dim=0)
	inputImageGraySegments = pt.cat(inputImageGraySegmentsList, dim=0)

	print("inputImageRGBSegments.shape = ", inputImageRGBSegments.shape)
	print("inputImageGraySegments.shape = ", inputImageGraySegments.shape)

	return inputImageRGBSegments, inputImageGraySegments

def applyRFfiltersList(resolutionProperties, inputImageRGBSegments, inputImageGraySegments, RFfiltersList, RFfiltersPropertiesList, ATORneuronListAllLayers):
	print("\tapplyRFfiltersList: resolutionIndex = ", resolutionProperties.resolutionIndex)

	ATORneuronListArray = ATORneuronListAllLayers[resolutionProperties.resolutionIndex]

	axesLengthMax, filterRadius, filterSize = ATORpt_RFfilter.getFilterDimensions(resolutionProperties)

	for RFlistIndex1 in range(len(RFfiltersPropertiesList)):
		print("RFlistIndex1 = ", RFlistIndex1)
		RFfiltersTensor = RFfiltersList[RFlistIndex1]
		RFfiltersPropertiesList2 = RFfiltersPropertiesList[RFlistIndex1]
		isColourFilter = RFfiltersPropertiesList2[0].isColourFilter
		numberOfDimensions = RFfiltersPropertiesList2[0].numberOfDimensions
		if isColourFilter:
			inputImageSegments = inputImageRGBSegments
		else:
			inputImageSegments = inputImageGraySegments

		filterApplicationResultThresholdIndicesList, filterApplicationResultThresholdedList, RFpropertiesList = ATORpt_RFgenerateApply.applyRFfilters(resolutionProperties, inputImageSegments, RFfiltersTensor, numberOfDimensions, RFfiltersPropertiesList2)

		print("ATORneuronList append: len(filterApplicationResultThresholdIndicesList) = ", len(filterApplicationResultThresholdIndicesList))
		for RFthresholdedListIndex, RFlistIndex in enumerate(filterApplicationResultThresholdIndicesList):

			filterApplicationResult = filterApplicationResultThresholdedList[RFthresholdedListIndex]
			RFproperties = RFpropertiesList[RFthresholdedListIndex]
			RFfilter = None
			RFImage = None
			if debugSaveRFfiltersAndImageSegments:
				RFfilter = RFfiltersTensor[RFproperties.filterIndex]
				RFImage = inputImageSegments[RFproperties.imageSegmentIndex]

			centerCoordinates = RFproperties.centerCoordinates
			allFilterCoordinatesWithinImageResult, imageSegmentStart, imageSegmentEnd = ATORpt_RFfilter.allFilterCoordinatesWithinImage(centerCoordinates, filterRadius, resolutionProperties.imageSize)
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
					ATORpt_RFgenerateApply.normaliseRFComponentWRTparent(resolutionProperties, neuron, parentNeuron.RFproperties)
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

		axesLengthMax, filterRadius, filterSize = ATORpt_RFfilter.getFilterDimensions(resolutionProperties)

		RFrangeGlobalMin, RFrangeGlobalMax = getRFrangeAtResolutionGlobal(resolutionPropertiesChild, neuronChild.RFproperties.centerCoordinates)
		RFrangeLocalMin, RFrangeLocalMax = getRFrangeAtResolutionLocal(resolutionProperties, RFrangeGlobalMin, RFrangeGlobalMax)
		for centerCoordinates1 in range(RFrangeLocalMin[0], RFrangeLocalMax[0] + 1, ellipseCenterCoordinatesResolution):
			for centerCoordinates2 in range(RFrangeLocalMin[1], RFrangeLocalMax[1] + 1, ellipseCenterCoordinatesResolution):
				centerCoordinates = [centerCoordinates1, centerCoordinates2]

				allFilterCoordinatesWithinImageResult, imageSegmentStart, imageSegmentEnd = ATORpt_RFfilter.allFilterCoordinatesWithinImage(centerCoordinates, filterRadius, resolutionProperties.imageSize)
				if allFilterCoordinatesWithinImageResult:

					ATORneuronList = ATORneuronListArray[centerCoordinates[0]][centerCoordinates[1]]
					if ATORneuronList is not None:
						for neuron in ATORneuronList:
							if ATORpt_RFgenerateApply.childRFoverlapsParentRF(neuronChild.RFpropertiesNormalisedGlobal, neuron.RFpropertiesNormalisedGlobal):
								foundParentNeuron = True
								parentNeuron = neuron

		if not foundParentNeuron:
			foundParentNeuron, neuronFound = findParentNeuron(ATORneuronListAllLayers, resolutionIndex, resolutionPropertiesChild, neuronChild)

	return foundParentNeuron, parentNeuron

def getRFrangeAtResolutionGlobal(resolutionProperties, centerCoordinates):
	axesLengthMax, filterRadius, filterSize = ATORpt_RFfilter.getFilterDimensions(resolutionProperties)

	RFrangeGlobalCentre = [int(centerCoordinates[0] * resolutionProperties.resolutionFactor), int(centerCoordinates[1] * resolutionProperties.resolutionFactor)]
	RFrangeGlobalSize = [int(filterSize[0] * resolutionProperties.resolutionFactor), int(filterSize[1] * resolutionProperties.resolutionFactor)]
	RFrangeGlobalMin = [RFrangeGlobalCentre[0] - RFrangeGlobalSize[0], RFrangeGlobalCentre[1] - RFrangeGlobalSize[1]]
	RFrangeGlobalMax = [RFrangeGlobalCentre[0] + RFrangeGlobalSize[0], RFrangeGlobalCentre[1] + RFrangeGlobalSize[1]]
	return RFrangeGlobalMin, RFrangeGlobalMax

def getRFrangeAtResolutionLocal(resolutionProperties, RFrangeGlobalMin, RFrangeGlobalMax):
	RFrangeLocalMin = [int(RFrangeGlobalMin[0] / resolutionProperties.resolutionFactor), int(RFrangeGlobalMin[1] / resolutionProperties.resolutionFactor)]
	RFrangeLocalMax = [int(RFrangeGlobalMax[0] / resolutionProperties.resolutionFactor), int(RFrangeGlobalMax[1] / resolutionProperties.resolutionFactor)]
	return RFrangeLocalMin, RFrangeLocalMax

@click.command()
@click.argument('inputimagefilename')

def main(inputimagefilename):
	#ATORpt_RFdetectEllipses.main(inputimagefilename)
	RFfiltersListAllRes, RFfiltersPropertiesListAllRes, ATORneuronListAllLayers = prepareRFhierarchyAccelerated()
	updateRFhierarchyAccelerated(RFfiltersListAllRes, RFfiltersPropertiesListAllRes, ATORneuronListAllLayers, inputimagefilename)	#trial image

if __name__ == "__main__":
	main()
