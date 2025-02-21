"""ATORpt_RFapplyCNN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORpt_RFmainFT.py

# Usage:
See ATORpt_RFmainFT.py

# Description:
ATORpt RF CNN

"""

import torch as pt
import numpy as np
import cv2
import copy
import torch.nn as nn

from ATORpt_RFglobalDefs import *


def generateCNNfilters(RFfiltersFeatureTypeList, RFfiltersPropertiesFeatureTypeList, resolutionProperties):
	imageSize = resolutionProperties.imageSize
	print("imageSize = ", imageSize)
	RFfiltersFeatureTypeConvLayerList = []
	#note RFfilters in RFfiltersList do not depend on resolutionIndex (resolutionProperties)
	for featureTypeIndex in range(len(RFfiltersFeatureTypeList)):
		RFfiltersList = RFfiltersFeatureTypeList[featureTypeIndex]
		RFfiltersPropertiesList = RFfiltersPropertiesFeatureTypeList[featureTypeIndex]
		RFfiltersFeatureTypeNumberKernels = 0
		RFfiltersFeatureKernelSize = 0
		if(RFuseParallelProcessedCNNRFchannelsImplementation==1):
			numberOfKernels = 0
		else:
			RFfiltersConvLayerList = []
		for RFfiltersTensorIndex in range(len(RFfiltersList)):
			RFfiltersTensor = RFfiltersList[RFfiltersTensorIndex]
			RFfiltersPropertiesTensor = RFfiltersPropertiesList[RFfiltersTensorIndex] 
			numberOfKernels = RFfiltersTensor.shape[0]
			isColourFilter = RFfiltersPropertiesTensor[0].isColourFilter
			if isColourFilter:
				print("isColourFilter")
				RFfilterNumberRGBchannels = rgbNumChannels
			else:
				print("!isColourFilter")
				RFfilterNumberRGBchannels = 1
			if(RFuseParallelProcessedCNNRFchannelsImplementation==2):
				RFfiltersTensorMerged = pt.reshape(RFfiltersTensor, (RFfiltersTensor.shape[0]*RFfiltersTensor.shape[3], RFfiltersTensor.shape[1], RFfiltersTensor.shape[2]))	#ensure filter C dim (RGB) is merged with filter index dim
			for RFfilterIndex in range(numberOfKernels):
				RFfiltersPropertiesTensor[RFfilterIndex].numberOfKernels = numberOfKernels
				RFfilter = RFfiltersTensor[RFfilterIndex]
				kernelSize = RFfilter.shape[0]
				if(RFuseParallelProcessedCNNRFchannelsImplementation==3):
					RFfiltersTensorMerged = pt.reshape(RFfilter, (RFfilter.shape[2], RFfilter.shape[0], RFfilter.shape[0]))	#ensure C dim is first
					RFfiltersConvLayer = nn.Conv2d(in_channels=RFfilterNumberRGBchannels, out_channels=RFfilterNumberRGBchannels, kernel_size=kernelSize, stride=calculateCNNstride(), padding='same', groups=RFfilterNumberRGBchannels)
					with pt.no_grad():
						print("RFfiltersConvLayer.weight.shape = ", RFfiltersConvLayer.weight.shape)
						print("RFfiltersTensorMerged.unsqueeze(1).shape = ", RFfiltersTensorMerged.unsqueeze(1).shape)
						RFfiltersConvLayer.weight = nn.Parameter(RFfiltersTensorMerged.unsqueeze(1))
					RFfiltersConvLayerList.append(RFfiltersConvLayer)
			RFfiltersFeatureTypeNumberKernels += numberOfKernels
			RFfiltersFeatureKernelSize = kernelSize
			if(RFuseParallelProcessedCNNRFchannelsImplementation==2):
				RFfiltersConvLayer = nn.Conv2d(in_channels=RFfilterNumberRGBchannels, out_channels=numberOfKernels*RFfilterNumberRGBchannels, kernel_size=kernelSize, stride=calculateCNNstride(), padding='same', groups=RFfilterNumberRGBchannels)
				with pt.no_grad():
					print("RFfiltersConvLayer.weight.shape = ", RFfiltersConvLayer.weight.shape)
					print("RFfiltersTensorMerged.unsqueeze(1).shape = ", RFfiltersTensorMerged.unsqueeze(1).shape)
					RFfiltersConvLayer.weight = nn.Parameter(RFfiltersTensorMerged.unsqueeze(1))
				RFfiltersConvLayerList.append(RFfiltersConvLayer)
		if(RFuseParallelProcessedCNNRFchannelsImplementation==1):
			RFfiltersConvLayer = nn.Conv2d(in_channels=RFfilterNumberRGBchannels, out_channels=RFfiltersFeatureTypeNumberKernels*RFfilterNumberRGBchannels, kernel_size=RFfiltersFeatureKernelSize, stride=calculateCNNstride(), padding='same', groups=RFfilterNumberRGBchannels)
			RFfiltersFeatureTypeConvLayerList.append(RFfiltersConvLayer)
			printe("generateCNNfilters error: RFuseParallelProcessedCNNRFchannelsImplementation==1 not supported as rgb channel dimensions will differ between RFfiltersTensor in RFfiltersList")
			#RFfiltersConvLayer.weight = nn.Parameter(RFfiltersFeatureTypeTensor)
		else:
			RFfiltersFeatureTypeConvLayerList.append(RFfiltersConvLayerList)
	return RFfiltersFeatureTypeConvLayerList

def applyCNNfilters(inputImage, RFfiltersConv, isColourFilter, numberOfKernels):
	assert RFuseParallelProcessedCNNRFchannelsImplementation == 2
	if isColourFilter:
		RFfilterNumberRGBchannels = rgbNumChannels
	else:
		RFfilterNumberRGBchannels = 1
	#print("inputImage.shape = ", inputImage.shape)
		#inputImage = inputImage.repeat(1, numberOfKernels, 1, 1)
		#print("inputImage.shape = ", inputImage.shape)
	filterApplicationResult = RFfiltersConv(inputImage)	#dim: batch_size, numberOfChannels, height, width
	#print("filterApplicationResult.shape = ", filterApplicationResult.shape)
	filterApplicationResult = pt.squeeze(filterApplicationResult, dim=0)	#dim: numberOfChannels, height, width
	#print("filterApplicationResult.shape = ", filterApplicationResult.shape)
	filterApplicationResult = pt.reshape(filterApplicationResult, (RFfilterNumberRGBchannels, filterApplicationResult.shape[0]//RFfilterNumberRGBchannels, filterApplicationResult.shape[1], filterApplicationResult.shape[2]))	#dim: rgbNumChannels, numberOfKernels, height, width
	filterApplicationResult = pt.sum(filterApplicationResult, dim=0)	#sum across rgb channels (same for grayscale images)	#dim: num_kernels, height, width
	filterApplicationResult = pt.permute(filterApplicationResult, (1, 2, 0))	#dim: height, width, numberOfKernels
	#print("filterApplicationResult.shape = ", filterApplicationResult.shape)
	filterApplicationResult = pt.reshape(filterApplicationResult, (filterApplicationResult.shape[0]*filterApplicationResult.shape[1], filterApplicationResult.shape[2]))	#place imageSegmentIndex at dim=0	#dim: numberOfImageSegments, numberOfKernels
	#print("filterApplicationResult.shape = ", filterApplicationResult.shape)
	filterApplicationResult = pt.reshape(filterApplicationResult, (filterApplicationResult.shape[0]*filterApplicationResult.shape[1],))	#dim: numberOfImageSegments*numberOfKernels
	#print("filterApplicationResult.shape = ", filterApplicationResult.shape)
	return filterApplicationResult

'''	
def calculateCNNpadding(kernelSize):
	print("kernelSize = ", kernelSize)
	paddingSize = (kernelSize-1)//2	#assume kernel size even (see ATORpt_RFpropertiesClass.getFilterDimensions)
	print("paddingSize = ", paddingSize)
	return kernelSize
	#CHECKTHIS; align with generateImageSegments (ie inputImageSize == outputImageSize) 
'''

def calculateCNNstride():
	return ellipseCenterCoordinatesResolution
