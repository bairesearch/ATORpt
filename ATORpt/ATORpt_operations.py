"""ATORpt_operations.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATORpt_main.py

# Description:
ATORpt operations

"""

import torch as pt
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

from ATORpt_globalDefs import *

if(debugGeometricHashingHardcoded):
	import matplotlib.pyplot as plt
	import matplotlib.cm as cm


def printKeypoints(keypointCoordinates):
	if(debugGeometricHashingHardcoded):
		print("printKeypoints")
		keypointCoordinatesCombined = pt.reshape(keypointCoordinates, (keypointCoordinates.shape[0]*keypointCoordinates.shape[1], keypointCoordinates.shape[2]))	#combine keyPointA/keyPointB/keyPointC
		keypointValuesCombined = pt.ones(keypointCoordinatesCombined[:, xAxis].shape)
		#print("keypointCoordinatesCombined.shape = ", keypointCoordinatesCombined.shape)
		#print("keypointValuesCombined.shape = ", keypointValuesCombined.shape)
		printImageCoordinates(keypointCoordinatesCombined[:, xAxis], keypointCoordinatesCombined[:, yAxis], keypointValuesCombined)

def printKeypointsIndex(keypointCoordinates, index):
	if(debugGeometricHashingHardcoded):
		print("printKeypointsIndex")
		#print("keypointCoordinates = ", keypointCoordinates)
		keypointCoordinatesCombined = keypointCoordinates[index, :, :]
		keypointValuesCombined = pt.ones(keypointCoordinatesCombined[:, xAxis].shape)
		#print("keypointCoordinatesCombined = ", keypointCoordinatesCombined)
		#print("keypointValuesCombined = ", keypointValuesCombined)
		printImageCoordinates(keypointCoordinatesCombined[:, xAxis], keypointCoordinatesCombined[:, yAxis], keypointValuesCombined[:])
					
def printPixelCoordinates(pixelCoordinates, pixelValues):
	if(debugGeometricHashingHardcoded):			
		print("printPixelCoordinates")
		#print("pixelCoordinates.shape = ", pixelCoordinates.shape)
		#print("pixelValues.shape = ", pixelValues.shape)
		printImageCoordinates(pixelCoordinates[:, xAxis], pixelCoordinates[:, yAxis], pixelValues)

def printPixelCoordinatesIndex(pixelCoordinates, pixelValues, index, text=None):
	if(debugGeometricHashingHardcoded):			
		print("printPixelCoordinatesIndex: " + text)
		printImageCoordinates(pixelCoordinates[index, xAxis], pixelCoordinates[index, yAxis], pixelValues[index])
					
def printFeatureMap(posEmbeddings, featureMapN):
	if(debugGeometricHashingHardcoded):
		print("printFeatureMap")
		printImageCoordinates(posEmbeddings[:, xAxis], posEmbeddings[:, yAxis], featureMapN)

def printPixelMap(posEmbeddings, tokens):
	if(debugGeometricHashingHardcoded):
		print("printPixelMap")
		firstIndexInBatch = 0
		printImageCoordinates(posEmbeddings[:, xAxis], posEmbeddings[:, yAxis], tokens[firstIndexInBatch])

def printImageCoordinates(x, y, values):

	#print("x.shape = ", x.shape)
	#print("y.shape = ", y.shape)
	#print("values.shape = ", values.shape)
				
	plotX = x.cpu().detach().numpy()
	plotY = y.cpu().detach().numpy()
	plotZ = values.cpu().detach().numpy()
	plotZ = 1.0-plotZ	#invert such that MNIST number pixels are displayed as black (on white background)

	markerSize = 2
	plt.subplot(121)
	plt.scatter(x=plotX, y=plotY, c=plotZ, s=markerSize, vmin=0, vmax=1, cmap=cm.gray)	#assume input is normalised (0->1.0) #unnormalised (0 -> 255)
	
	plt.xlim(-1, 1)
	plt.ylim(-1, 1)
	plt.gca().set_aspect('equal', adjustable='box')
	
	plt.show()		
			
def getInputLayerNumTokens(numberOfPatches):
	inputLayerNumTokens = int(numberOfPatches ** 2)
	return inputLayerNumTokens

def getHiddenLayerNumTokens(numberOfPatches):
	hiddenLayerNumTokens = numberOfPatches ** 2 + 1
	return hiddenLayerNumTokens

def getInputDim(inputShape, patchSize):
	numberOfChannels = inputShape[0]
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
	
def getPositionalEmbeddingsAbsolute(numberOfPatches):
	inputLayerNumTokens = getInputLayerNumTokens(numberOfPatches)
	#see createLinearPatches specification; patches[imageIndex, i*numberOfPatches + j]
	posEmbeddingsAbsolute = pt.zeros(inputLayerNumTokens, 2)  #pos embeddings absolute include x/y dim only
	posEmbeddingsAbsolute[:, yAxis] = pt.unsqueeze(pt.arange(1, numberOfPatches+1),1).repeat(1, numberOfPatches).flatten()
	posEmbeddingsAbsolute[:, xAxis] = pt.arange(1, numberOfPatches+1).repeat(numberOfPatches)

	return posEmbeddingsAbsolute

def createLinearPatches(images, numberOfPatches, flattenChannels=True):
	batchSize, numberOfChannels, imageHeight, imageWidth = images.shape
	inputShape = (numberOfChannels, imageHeight, imageWidth)
 
	if(imageHeight != imageWidth):
		print("createLinearPatches requires imageHeight == imageWidth")

	if(flattenChannels):
		patches = pt.zeros(batchSize, getInputLayerNumTokens(numberOfPatches), getPatchSizeFlat(inputShape, numberOfPatches))
	else:
		patches = pt.zeros(batchSize, numberOfChannels, getInputLayerNumTokens(numberOfPatches), getPatchSizeFlat(inputShape, numberOfPatches))

	patchLength = getPatchLength(inputShape, numberOfPatches)

	for imageIndex, image in enumerate(images):
		for i in range(numberOfPatches):	#y axis
			for j in range(numberOfPatches):	#x axis
				patch = image[:, i*patchLength:(i+1)*patchLength, j*patchLength:(j+1)*patchLength]
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
		for i in range(numberOfPatches):
			for j in range(numberOfPatches):
				for k in range(numberOfGeometricDimensions):
					images[imageIndex, i, j, k] = sequence[i*numberOfPatches + j, k]

	return images
