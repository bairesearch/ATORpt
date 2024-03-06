"""ATORpt_RFmain.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORpt_main.py

# Usage:
source activate pytorchsenv2
python ATORpt_RFmain.py images/leaf1.png

# Description:
ATORpt RF is a receptive field implementation for ATOR feature detection (ellipse centroids and tri corners)

ATORpt RF supports ellipsoid features (for centroid detection), and normalises them with respect to their major/minor ellipticity axis orientation. 
There are a number of advantages of using ellipsoid features over point features;
* the number of feature sets/normalised snapshots required is significantly reduced
* scene component structure can be maintained (as detected component ellipses can be represented in a hierarchical graph structure)
* features can still be detected where there are no point features available
Ellipse features/components are detected based on simulated artificial receptive fields; RF (on/off, off/on).

ATORpt will also support point (corner/centroid) features of the ATOR specification using a third party library; 
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
import torch as pt
import torch.nn.functional as F

from ATORpt_RFglobalDefs import *
import ATORpt_RFdetectEllipses
import ATORpt_RFgenerateApply
import ATORpt_RFproperties
import ATORpt_RFfilter
import ATORpt_RFoperations
import ATORpt_RFellipse
import ATORpt_RFtri
if(RFuseParallelProcessedCNN):
	import ATORpt_RFCNN

resolutionIndexFirst = 0

class ATORneuronClass():
	def __init__(self, resolutionProperties, RFproperties, RFfilter, RFImage):
		self.resolutionIndex = resolutionProperties.resolutionIndex
		self.RFproperties = RFproperties
		self.RFpropertiesNormalised = ATORpt_RFgenerateApply.normaliseLocalRFproperties(RFproperties)
		self.RFpropertiesNormalisedWRTparent = None
		self.RFpropertiesNormalisedGlobal = ATORpt_RFgenerateApply.normaliseGlobalRFproperties(RFproperties, resolutionProperties.resolutionFactor)
		if RFsaveRFfiltersAndImageSegments:
			self.RFfilter = RFfilter
			self.RFfilterNormalised = ATORpt_RFfilter.normaliseRFfilter(RFfilter, RFproperties)
			self.RFfilterNormalisedWRTparent = None
			self.RFImage = RFImage
			if(RFsaveRFimageSegments):
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
		resolutionProperties = ATORpt_RFoperations.RFresolutionProperties(resolutionIndex, resolutionIndexFirst, numberOfResolutions, imageSizeBase)
		RFfiltersList, RFfiltersPropertiesList = ATORpt_RFgenerateApply.generateRFfilters(resolutionProperties, generateRFfiltersEllipse, generateRFfiltersTri)
		RFfiltersListAllRes.append(RFfiltersList)
		RFfiltersPropertiesListAllRes.append(RFfiltersPropertiesList)
	
	for resolutionIndex in range(resolutionIndexFirst, resolutionIndexMax):
		resolutionProperties = ATORpt_RFoperations.RFresolutionProperties(resolutionIndex, resolutionIndexFirst, numberOfResolutions, imageSizeBase)
		ATORneuronListArray = initialiseATORneuronListArray(resolutionProperties)
		ATORneuronListAllLayers.append(ATORneuronListArray)
	
	return RFfiltersListAllRes, RFfiltersPropertiesListAllRes, ATORneuronListAllLayers
	
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
			RFfiltersFeatureTypeConvLayerList = ATORpt_RFCNN.generateCNNfilters(RFfiltersListAllRes[resolutionIndex], RFfiltersPropertiesListAllRes[resolutionIndex], resolutionProperties)
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

def initialiseATORneuronListArray(resolutionProperties):
	size = (resolutionProperties.imageSize[0], resolutionProperties.imageSize[1])
	ATORneuronListArray = [[None for _ in range(size[1])] for _ in range(size[0])]
	return ATORneuronListArray

if(RFuseParallelProcessedCNN):
	def generateImageSegment(resolutionProperties, RFproperties, inputImageTensor, isColourFilter):
		filterRadius = RFproperties.imageSize[0]
		centerCoordinates = (RFproperties.imageSegmentIndex//resolutionProperties.imageSize[1], RFproperties.imageSegmentIndex%resolutionProperties.imageSize[1])
		inputImageSegment = generateImageSegmentBand(resolutionProperties, centerCoordinates, filterRadius, inputImageTensor)
		inputImageSegment = inputImageSegment[0]
		inputImageSegment = inputImageSegment.permute(1, 2, 0)	#ensure channels dim is last #match RFfilter dims: height, width, channels	#see transformRFfilterTF2D
		return inputImageSegment
else:
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
				inputImageRGBSegment = generateImageSegmentBand(resolutionProperties, centerCoordinates, filterRadius, inputImageRGBTensor)
				inputImageGraySegment = generateImageSegmentBand(resolutionProperties, centerCoordinates, filterRadius, inputImageGrayTensor)
				inputImageRGBSegmentsList.append(inputImageRGBSegment)
				inputImageGraySegmentsList.append(inputImageGraySegment)
				imageSegmentIndex = imageSegmentIndex + 1

		inputImageRGBSegments = pt.cat(inputImageRGBSegmentsList, dim=0)
		inputImageGraySegments = pt.cat(inputImageGraySegmentsList, dim=0)

		return inputImageRGBSegments, inputImageGraySegments

def generateImageSegmentBand(resolutionProperties, centerCoordinates, filterRadius, inputImageTensor):
	allFilterCoordinatesWithinImageResult, imageSegmentStart, imageSegmentEnd = ATORpt_RFfilter.allFilterCoordinatesWithinImage(centerCoordinates, filterRadius, resolutionProperties.imageSize)
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
		#axesLengthMax, filterRadius, filterSize = ATORpt_RFfilter.getFilterDimensions(resolutionProperties)	#OLD
		if(RFfeatureTypeListIndex == RFfeatureTypeIndexEllipse):
			axesLengthMax, filterRadius, filterSize = ATORpt_RFellipse.getFilterDimensions(resolutionProperties)
		elif(RFfeatureTypeListIndex == RFfeatureTypeIndexTri):
			axesLengthMax, filterRadius, filterSize = ATORpt_RFtri.getFilterDimensions(resolutionProperties)
		
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
				RFpropertiesList = ATORpt_RFgenerateApply.applyRFfilters(resolutionProperties, RFfiltersTensor, numberOfDimensions, RFfiltersPropertiesList2, inputImage=inputImage, RFfiltersConv=RFfiltersConv)
			else:
				if isColourFilter:
					inputImageSegments = inputImageRGBSegments
				else:
					inputImageSegments = inputImageGraySegments
				RFpropertiesList = ATORpt_RFgenerateApply.applyRFfilters(resolutionProperties, RFfiltersTensor, numberOfDimensions, RFfiltersPropertiesList2, inputImageSegments=inputImageSegments)

			if(RFdetectTriFeaturesSeparately):
				if(RFfiltersPropertiesList2[0].RFtype == RFtypeTemporaryPointFeatureKernel):
					#print("RFpropertiesList = ", RFpropertiesList)
					RFpropertiesList = ATORpt_RFgenerateApply.generateRFtypeTriFromPointFeatureSets(resolutionProperties, RFpropertiesList)	#generate RFtypeTri for sets of features
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
