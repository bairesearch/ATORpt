"""ATORpt_E2EgeometricHashing.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt E2E geometric hashing

"""

import torch as pt
import torch.nn as nn

import torchvision.transforms as T

from ATORpt_globalDefs import *
import ATORpt_E2Eoperations
import ATORpt_E2Ekeypoints
import ATORpt_PTgeometricHashing
import ATORpt_operations

 
class GeometricHashingClass(nn.Module):
	def __init__(self, numberOfTokenDimensions, numberOfPatches):

		self.numberOfPatches = numberOfPatches
		self.numberOfTokenDimensions = numberOfTokenDimensions

		super(GeometricHashingClass, self).__init__()
		  
		self.cosSim = pt.nn.CosineSimilarity(dim=1)  #CHECKTHIS: dim=1

		self.numberOfGeometricDimensions = numberOfGeometricDimensions	#2D object data (2DOD)
		self.geometricHashingNumKeypoints = self.numberOfGeometricDimensions+1	#2DOD: 3: 3DOD: 4   #number of features to use to perform geometric hashing (depends on input object data dimensions; 2DOD/3DOD)
		self.geometricHashingNumPixels = 1  #1 pixel (token) will be transformed
		
		if(useGeometricHashingAMANN):
			self.geometricHashingNumberLayers = self.geometricHashingNumKeypoints #number of consecutive transformations required to be encoded/learnt by MSAgeomtricHashing
			if(useGeometricHashingProbabilisticKeypoints):
				if(useGeometricHashingProbabilisticKeypointsSoftMax):
					self.softmax = nn.Softmax(dim=-1)
				if(useGeometricHashingProbabilisticKeypointsNonlinearity):
					self.activationFunction = pt.nn.ReLU(inplace=False)
				
				inputLayerNumTokens = ATORpt_E2Eoperations.getInputLayerNumTokens(numberOfPatches)
				self.numberOfAttentionDimensions = 1
				self.geometricHashingNumInputs = inputLayerNumTokens + 1
				self.geometricHashingInputDim = (self.geometricHashingNumInputs*(self.numberOfGeometricDimensions+self.numberOfAttentionDimensions))
			else:  
				self.geometricHashingNumInputs = self.geometricHashingNumKeypoints+self.geometricHashingNumPixels
				self.geometricHashingInputDim = self.geometricHashingNumInputs * self.numberOfGeometricDimensions
	 
			linearAdditiveMultiplicativeList = []
			for i in range(self.geometricHashingNumberLayers):
				linearAdditiveMultiplicativeList.append(ATORpt_E2EAMANN.LayerAdditiveMultiplicativeClass(self.geometricHashingInputDim, self.geometricHashingInputDim, useMultiplicativeUnits=True))
			linearAdditiveMultiplicativeList.append(ATORpt_E2EAMANN.LayerAdditiveMultiplicativeClass(self.geometricHashingInputDim, self.numberOfGeometricDimensions, useMultiplicativeUnits=False))
			self.linearAdditiveMultiplicativeModuleList = nn.ModuleList(linearAdditiveMultiplicativeList)

	def forward(self, images, posEmbeddings, sequences, featureMap):

		posEmbeddingsGeometricNormalisedList = []
		batchSize = sequences.shape[0]
		sequenceLength = sequences.shape[1]

		#print("sequences.shape = ", sequences.shape)
		#print("featureMap.shape = ", featureMap.shape)
		
		#sequences shape = batchSize, sequenceLength, numberOfTokenDimensions
		for batchIndex in range(batchSize):

			print("batchIndex = ", batchIndex)
			
			imageN = images[batchIndex]
			pixelValuesN = sequences[batchIndex]
			featureMapN = featureMap[batchIndex]

			if(debugGeometricHashingParallel):
				ATORpt_E2Eoperations.printImage(imageN)
				#ATORpt_E2Eoperations.printFeatureMap(posEmbeddings, featureMapN)

			posEmbeddingsNormalised = ATORpt_E2Eoperations.normaliseInputs0to1(posEmbeddings, dim=0)	#normalise across sequenceLength dimension
			
			#print("posEmbeddings = ", posEmbeddings)
			#print("posEmbeddingsNormalised = ", posEmbeddingsNormalised)
			
			geometricHashingPixelPosEmbeddings = posEmbeddingsNormalised

			geometricHashingKeypointsPosEmbeddings, geometricHashingPixelPosEmbeddings = ATORpt_E2Ekeypoints.performKeypointDetection(self, featureMapN, posEmbeddings, posEmbeddingsNormalised, geometricHashingPixelPosEmbeddings)
			
			if(useGeometricHashingAMANN):
				posEmbeddingsAbsoluteGeoNormalisedN = self.performGeometricHashingAMANN(geometricHashingKeypointsPosEmbeddings, geometricHashingPixelPosEmbeddings)
			else:
				print("TODO: need to ensure that geometricHashingKeypointsPosEmbeddings and geometricHashingPixelPosEmbeddings are batched")
				print("\tgeometricHashingKeypointsPosEmbeddings.shape = ", geometricHashingKeypointsPosEmbeddings.shape)
				print("\tgeometricHashingPixelPosEmbeddings.shape = ", geometricHashingPixelPosEmbeddings.shape)
				print("TODO: need to ensure that geometricHashingKeypointsPosEmbeddings and geometricHashingPixelPosEmbeddings are in format xAxisGeometricHashing,yAxisGeometricHashing")
				posEmbeddingsAbsoluteGeoNormalisedN = ATORpt_PTgeometricHashing.performGeometricHashingParallel(geometricHashingKeypointsPosEmbeddings, geometricHashingPixelPosEmbeddings, pixelValuesN)

			posEmbeddingsGeometricNormalisedList.append(posEmbeddingsAbsoluteGeoNormalisedN)

		posEmbeddingsGeometricNormalised = pt.stack(posEmbeddingsGeometricNormalisedList, dim=0) 	#CHECKTHIS: normalise across sequenceLength dimension

		return posEmbeddingsGeometricNormalised

	def performGeometricHashingAMANN(self, geometricHashingKeypointsPosEmbeddings, geometricHashingPixelPosEmbeddings):
				
		geometricHashingKeypointsPosEmbeddings = geometricHashingKeypointsPosEmbeddings.flatten(start_dim=1, end_dim=2)

		geometricHashingInputs = pt.cat([geometricHashingKeypointsPosEmbeddings, geometricHashingPixelPosEmbeddings], dim=1)

		if(useGeometricHashingReduceInputMagnitude):
			geometricHashingInputs = geometricHashingInputs / 5.0

		geometricHashingLayer = geometricHashingInputs
		for i, l in enumerate(self.linearAdditiveMultiplicativeModuleList):
			geometricHashingLayer = l(geometricHashingLayer)
			#print("geometricHashingLayer = ", geometricHashingLayer)
		geometricHashingOutput = geometricHashingLayer
		#print("geometricHashingOutput = ", geometricHashingOutput)

		posEmbeddingsAbsoluteGeoNormalisedN = geometricHashingOutput

		if(useGeometricHashingNormaliseOutput):
			posEmbeddingsAbsoluteGeoNormalisedN = ATORpt_E2Eoperations.normaliseInputs0to1(posEmbeddingsAbsoluteGeoNormalisedN, dim=0)
		#print("posEmbeddingsAbsoluteGeoNormalisedN = ", posEmbeddingsAbsoluteGeoNormalisedN)
				
		return posEmbeddingsAbsoluteGeoNormalisedN
		
