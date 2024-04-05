"""ATORpt_E2Eoperations.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt E2E operations

"""

import torch as pt
import torch.nn as nn
import numpy as np
import ATORpt_operations

import torch.nn.functional as F

from ATORpt_globalDefs import *

def printImage(image)
	ATORpt_operations.printImage(image)

def getInputLayerNumTokens(numberOfPatches):
	inputLayerNumTokens = int(numberOfPatches ** 2)
	return inputLayerNumTokens

def getHiddenLayerNumTokens(numberOfPatches):
	hiddenLayerNumTokens = numberOfPatches ** 2 + 1
	return hiddenLayerNumTokens

def getInputDim(inputShape, patchSize):
	numberOfChannels = inputShape[0]
	numberOfInputDimensions = getInputDim2(numberOfChannels, patchSize)
	return numberOfInputDimensions

def getInputDim2(numberOfChannels, patchSize):
	numberOfInputDimensions = int(numberOfChannels * getPatchSizeFlat2(patchSize))
	return numberOfInputDimensions

def getPatchSize(inputShape, numberOfPatches):
	patchSize = (inputShape[1]//numberOfPatches, inputShape[2]//numberOfPatches)
	return patchSize

def getPatchLength(inputShape, numberOfPatches):
	(numberOfChannels, imageHeight, imageWidth) = inputShape
	patchLength = imageHeight // numberOfPatches
	return patchLength

def getPatchSizeFlat(inputShape, numberOfPatches):
	patchSize = getPatchSize(inputShape, numberOfPatches)
	patchSizeFlat = getPatchSizeFlat2(patchSize)
	return patchSizeFlat

def getPatchSizeFlat2(patchSize):
	patchSizeFlat = patchSize[0]*patchSize[1]
	return patchSizeFlat

def normaliseInputs0to1(A, dim=None):
	if(useGeometricHashingKeypointNormalisation):
		if(dim is None):
			A = A / A.max()
		else:
			A = A / A.amax(dim=dim, keepdim=True)
		#A = F.normalize(A, dim)
	return A

def getPositionalEmbeddings(sequenceLength, d):
	result = pt.ones(sequenceLength, d)
	for i in range(sequenceLength):
		for j in range(d):
			result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
	return result

def getPositionalEmbeddingsAbsolute(numberOfPatches, xAxis, yAxis):
	inputLayerNumTokens = getInputLayerNumTokens(numberOfPatches)
	#see createLinearPatches specification; patches[imageIndex, i*numberOfPatches + j]
	posEmbeddingsAbsolute = pt.zeros(inputLayerNumTokens, 2)  #pos embeddings absolute include x/y dim only
	posEmbeddingsAbsolute[:, xAxis] = pt.arange(1, numberOfPatches+1).repeat(numberOfPatches)
	posEmbeddingsAbsolute[:, yAxis] = pt.unsqueeze(pt.arange(1, numberOfPatches+1),1).repeat(1, numberOfPatches).flatten()
	return posEmbeddingsAbsolute

'''
if(useATORCPPserial):
	def createLinearPatches(patches, flattenChannels=True):
		numberOfPatches, numberOfChannels, imageHeight, imageWidth = patches.shape
		if(flattenChannels):
			patches = patches.reshape(numberOfPatches, numberOfChannels*imageHeight*imageWidth)
		else:
			patches = patches.reshape(numberOfPatches, numberOfChannels, imageHeight*imageWidth)
		return patches
'''

def createLinearPatches(images, numberOfPatches, isATORmodel, flattenChannels=True):
	'''
	#not required as imageHeight and imageWidth must be identical
	if(isATORmodel):
		batchSize, numberOfChannels, imageWidth, imageHeight = images.shape
	else:
		batchSize, numberOfChannels, imageHeight, imageWidth = images.shape
	'''
	batchSize, numberOfChannels, imageWidth, imageHeight = images.shape
	inputShape = (numberOfChannels, imageHeight, imageWidth)

	if(imageHeight != imageWidth):
		print("createLinearPatches requires imageHeight == imageWidth")

	if(flattenChannels):
		patches = pt.zeros(batchSize, getInputLayerNumTokens(numberOfPatches), getPatchSizeFlat(inputShape, numberOfPatches))
	else:
		patches = pt.zeros(batchSize, numberOfChannels, getInputLayerNumTokens(numberOfPatches), getPatchSizeFlat(inputShape, numberOfPatches))

	patchLength = getPatchLength(inputShape, numberOfPatches)

	for imageIndex, image in enumerate(images):
		for i in range(numberOfPatches):	#!isATORmodel: yAxis, isATORmodel: xAxis
			for j in range(numberOfPatches):	#!isATORmodel: yAxis, isATORmodel: xAxis
				patch = image[:, i*patchLength:(i+1)*patchLength, j*patchLength:(j+1)*patchLength]
				'''
				#not required as customViT should be agnostic to subpatch dimensions once transformed via ATOR
				if(not isATORmodel):
					if(xAxisGeometricHashing == 0):
						patch = patch.permute(0, 2, 1)	#numberOfChannels, imageWidth, imageHeight
					elif(xAxisGeometricHashing == 1):
						pass #patch = patch.permute(0, 1, 2)	#numberOfChannels, imageHeight, imageWidth
				'''
				if(flattenChannels):
					patch = patch.flatten()
					patches[imageIndex, i*numberOfPatches + j] = patch
				else:
					patch = patch.flatten(start_dim=1, end_dim=2)
					for k in range(numberOfChannels):
						patches[imageIndex, k, i*numberOfPatches + j] = patch[k]
	return patches

def uncreateLinearPatches(sequences, numberOfPatches, numberOfGeoDimensions):
	batchSize = sequences.shape[0]
	seqLength = sequences.shape[1]  #numberOfPatches*numberOfPatches

	images = pt.zeros(batchSize, numberOfPatches, numberOfPatches, numberOfGeoDimensions)

	for imageIndex, sequence in enumerate(sequences):
		for i in range(numberOfPatches):	#y axis	[yAxisViT]
			for j in range(numberOfPatches):	#x axis	[xAxisViT]
				for k in range(numberOfGeometricDimensions2DOD):
					images[imageIndex, i, j, k] = sequence[i*numberOfPatches + j, k]

	return images
