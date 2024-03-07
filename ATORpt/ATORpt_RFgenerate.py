"""ATORpt_RFgenerate.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt RF generate

"""

import torch as pt
import torch
import torch.nn.functional as F
import numpy as np
import copy
import math

from ATORpt_RFglobalDefs import *
import ATORpt_RFgenerateEllipse
import ATORpt_RFgenerateTri
import ATORpt_RFoperations

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
		RFfiltersList, RFfiltersPropertiesList = generateRFfilters(resolutionProperties, generateRFfiltersEllipse, generateRFfiltersTri)
		RFfiltersListAllRes.append(RFfiltersList)
		RFfiltersPropertiesListAllRes.append(RFfiltersPropertiesList)
	
	for resolutionIndex in range(resolutionIndexFirst, resolutionIndexMax):
		resolutionProperties = ATORpt_RFoperations.RFresolutionProperties(resolutionIndex, resolutionIndexFirst, numberOfResolutions, imageSizeBase)
		ATORneuronListArray = initialiseATORneuronListArray(resolutionProperties)
		ATORneuronListAllLayers.append(ATORneuronListArray)
	
	return RFfiltersListAllRes, RFfiltersPropertiesListAllRes, ATORneuronListAllLayers

def initialiseATORneuronListArray(resolutionProperties):
	size = (resolutionProperties.imageSize[0], resolutionProperties.imageSize[1])
	ATORneuronListArray = [[None for _ in range(size[1])] for _ in range(size[0])]
	return ATORneuronListArray
	
def generateRFfilters(resolutionProperties, generateRFfiltersEllipse, generateRFfiltersTri):
	RFfiltersFeatureTypeList = []
	RFfiltersPropertiesFeatureTypeList = []
	if generateRFfiltersEllipse:
		RFfiltersList, RFfiltersPropertiesList = ATORpt_RFgenerateEllipse.generateRFfiltersEllipse(resolutionProperties)
		RFfiltersFeatureTypeList.append(RFfiltersList)
		RFfiltersPropertiesFeatureTypeList.append(RFfiltersPropertiesList)
	if generateRFfiltersTri:
		RFfiltersList, RFfiltersPropertiesList = ATORpt_RFgenerateTri.generateRFfiltersTri(resolutionProperties)
		RFfiltersFeatureTypeList.append(RFfiltersList)
		RFfiltersPropertiesFeatureTypeList.append(RFfiltersPropertiesList)
	return RFfiltersFeatureTypeList, RFfiltersPropertiesFeatureTypeList

