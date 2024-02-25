"""ATORpt_geometricHashing.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATORpt_main.py

# Description:
ATORpt geometric hashing

"""

import torch as pt
import torch.nn as nn

import torchvision.transforms as T

from ATORpt_globalDefs import *
import ATORpt_operations
import ATORpt_featureDetector

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
				
				inputLayerNumTokens = ATORpt_operations.getInputLayerNumTokens(numberOfPatches)
				self.numberOfAttentionDimensions = 1
				self.geometricHashingNumInputs = inputLayerNumTokens + 1
				self.geometricHashingInputDim = (self.geometricHashingNumInputs*(self.numberOfGeometricDimensions+self.numberOfAttentionDimensions))
			else:  
				self.geometricHashingNumInputs = self.geometricHashingNumKeypoints+self.geometricHashingNumPixels
				self.geometricHashingInputDim = self.geometricHashingNumInputs * self.numberOfGeometricDimensions
	 
			linearAdditiveMultiplicativeList = []
			for i in range(self.geometricHashingNumberLayers):
				linearAdditiveMultiplicativeList.append(ATORpt_AMANN.LayerAdditiveMultiplicativeClass(self.geometricHashingInputDim, self.geometricHashingInputDim, useMultiplicativeUnits=True))
			linearAdditiveMultiplicativeList.append(ATORpt_AMANN.LayerAdditiveMultiplicativeClass(self.geometricHashingInputDim, self.numberOfGeometricDimensions, useMultiplicativeUnits=False))
			self.linearAdditiveMultiplicativeModuleList = nn.ModuleList(linearAdditiveMultiplicativeList)


	def offsetReLU(self, Z):
		Z = Z - useGeometricHashingProbabilisticKeypointsNonlinearityOffset
		A = self.activationFunction(Z)
		return A

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

			if(debugGeometricHashingHardcoded):
				ATORpt_operations.printImage(imageN)
				#ATORpt_operations.printFeatureMap(posEmbeddings, featureMapN)

			posEmbeddingsNormalised = ATORpt_operations.normaliseInputs0to1(posEmbeddings, dim=0)	#normalise across sequenceLength dimension
			
			#print("posEmbeddings = ", posEmbeddings)
			#print("posEmbeddingsNormalised = ", posEmbeddingsNormalised)
			
			geometricHashingPixelPosEmbeddings = posEmbeddingsNormalised

			geometricHashingKeypointsPosEmbeddings, geometricHashingPixelPosEmbeddings = ATORpt_featureDetector.performKeypointDetection(self, featureMapN, posEmbeddings, posEmbeddingsNormalised, geometricHashingPixelPosEmbeddings)
			
			if(useGeometricHashingAMANN):
				posEmbeddingsAbsoluteGeoNormalisedN = self.performGeometricHashingAMANN(geometricHashingKeypointsPosEmbeddings, geometricHashingPixelPosEmbeddings)
			else:
				posEmbeddingsAbsoluteGeoNormalisedN = self.performGeometricHashingHardcoded(geometricHashingKeypointsPosEmbeddings, geometricHashingPixelPosEmbeddings, pixelValuesN)

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
			posEmbeddingsAbsoluteGeoNormalisedN = ATORpt_operations.normaliseInputs0to1(posEmbeddingsAbsoluteGeoNormalisedN, dim=0)
		#print("posEmbeddingsAbsoluteGeoNormalisedN = ", posEmbeddingsAbsoluteGeoNormalisedN)
				
		return posEmbeddingsAbsoluteGeoNormalisedN
		
	def performGeometricHashingHardcoded(self, geometricHashingKeypointsPosEmbeddings, geometricHashingPixelPosEmbeddings, pixelValues=None):
		#based on https://patentscope.wipo.int/search/en/detail.jsf?docId=WO2011088497 Fig 30->35
		#see ATORmethod2DOD:transformObjectData2DOD for unvectorised method
			
		#implementation limitation: assume input image is roughly upright; only perform 1 geometric hashing transformation (not geometricHashingNumKeypoints transformations, based on every possible permutation of keypoints)
		keypointCoordinates = geometricHashingKeypointsPosEmbeddings
		pixelCoordinates = geometricHashingPixelPosEmbeddings

		
		if(debugGeometricHashingHardcoded):
			#artificially set position of first pixel and first set of keypoints to good combination for visualisation
			firstSequenceIndex = 0
			pixelCoordinates[firstSequenceIndex][xAxis] = 0.2
			pixelCoordinates[firstSequenceIndex][yAxis] = 0.6
			pixelValues[firstSequenceIndex] = 1.0
			keypointCoordinates[firstSequenceIndex][0][xAxis] = 0.7
			keypointCoordinates[firstSequenceIndex][0][yAxis] = 0.3
			keypointCoordinates[firstSequenceIndex][1][xAxis] = 0.4
			keypointCoordinates[firstSequenceIndex][1][yAxis] = 0.2
			keypointCoordinates[firstSequenceIndex][2][xAxis] = 0.1
			keypointCoordinates[firstSequenceIndex][2][yAxis] = 0.8
		
		#ATORpt_operations.printPixelCoordinates(pixelCoordinates, pixelValues)
		#ATORpt_operations.printKeypoints(keypointCoordinates)
		ATORpt_operations.printPixelCoordinatesIndex(pixelCoordinates, pixelValues, index=0, text="step0")
		ATORpt_operations.printKeypointsIndex(keypointCoordinates, index=0)
		#print("0 pixelCoordinates = ", pixelCoordinates)
		
		#reorder keypointCoordinates;
		# kp2
		#  \\
		#   \_\
		#  kp1 kp0  

		keypointCoordinates = self.keypointSwap(keypointCoordinates, 2, 1, 0, yAxis)
		keypointCoordinates = self.keypointSwap(keypointCoordinates, 2, 0, 1, yAxis)
		keypointCoordinates = self.keypointSwap(keypointCoordinates, 0, 1, 2, xAxis)

		#apply hardcoded geometric hashing function;
		
		#print("pixelCoordinates.shape = ", pixelCoordinates.shape)
		#print("keypointCoordinates.shape = ", keypointCoordinates.shape)

		#step 1 (shift x/y - wrt centre of keypointCoordinates [0, 1]):
		#translate object data on X and Y axis such that the object triangle base is positioned at centre of keypointCoordinates [0, 1]):
		#Fig 31
		keypointsTriBaseCentre = pt.add(keypointCoordinates[:, 0], keypointCoordinates[:, 1])/2.0
		pixelCoordinates = pt.subtract(pixelCoordinates, keypointsTriBaseCentre)
		ATORpt_operations.printPixelCoordinatesIndex(pixelCoordinates, pixelValues, index=0, text="step1")
		#print("1 pixelCoordinates = ", pixelCoordinates)
		
		#step 2 (rotate - wrt keypointCoordinates [0, 1]):
		#2ia. rotate object data such that the object triangle side is parallel with X axis [and 2ii. third apex is above the lowest 2 apexes]
		#Fig 31
		keypointsTriBaseVec = pt.subtract(keypointCoordinates[:, 1], keypointCoordinates[:, 0])
		rotationMatrix = self.createRotationMatrix2Dvec(keypointsTriBaseVec)
		pixelCoordinates = self.applyRotation2D(pixelCoordinates, rotationMatrix)
		ATORpt_operations.printPixelCoordinatesIndex(pixelCoordinates, pixelValues, index=0, text="step2")
		#print("2 pixelCoordinates = ", pixelCoordinates)
		
		#step 3 (scale x - wrt keypointCoordinates [0, 1]):   
		#1a. Scale object data such that the object triangle side is of same length as a predefined side of a predefined triangle
		#Fig 33
		keypointsTriBaseSizeX = pt.subtract(keypointCoordinates[:, 0, xAxis], keypointCoordinates[:, 1, xAxis])
		pixelsX = pixelCoordinates[:, xAxis] 
		pixelsX = pt.divide(pixelsX, keypointsTriBaseSizeX)
		pixelCoordinates[:, xAxis] = pixelsX
		ATORpt_operations.printPixelCoordinatesIndex(pixelCoordinates, pixelValues, index=0, text="step3")
		#print("3 pixelCoordinates = ", pixelCoordinates)
		
		#step 4 (scale y - wrt keypointCoordinates [1y, 2y]):
		#3a. Scale object data on Y axis such that the third apex is the same perpendicular distance away from the side as is the case for the predefined triangle.
		#Fig 34
		keypointsTriHeightSize = pt.subtract(keypointCoordinates[:, 2, yAxis], keypointCoordinates[:, 1, yAxis])
		pixelsY = pixelCoordinates[:, yAxis]
		pixelsY = pt.divide(pixelsY, keypointsTriHeightSize)
		pixelCoordinates[:, yAxis] = pixelsY
		ATORpt_operations.printPixelCoordinatesIndex(pixelCoordinates, pixelValues, index=0, text="step4")
		#print("4 pixelCoordinates = ", pixelCoordinates)
		
		#step 5 (shear):
		#4a. shear object data along X axis such that object triangle apexes are coincident with predefined triangle apexes
		#Fig 35
		#xAxisDistanceBetweenThirdApexOfObjectTriangleAndSideLeftApex = calculateXaxisDistanceBetweenThirdApexOfObjectTriangleAndSideLeftApex(transformedObjectTriangle, side)
		#shearRequired4a = (xAxisDistanceBetweenThirdApexOfObjectTriangleAndSideLeftApex - (lengthOfPredefinedTriangleSide/2))/perpendicularDistanceBetweenThirdApexOfPredefinedTriangleAndSide;		
		keypointsTriBaseCentreX = pt.add(keypointCoordinates[:, 0, xAxis], keypointCoordinates[:, 1, xAxis])/2.0
		keypointsTriTipVecX = pt.subtract(keypointCoordinates[:, 2, xAxis], keypointsTriBaseCentreX)	#CHECKTHIS
		keypointsTriTipVecY = pt.subtract(keypointCoordinates[:, 2, yAxis], keypointCoordinates[:, 0, yAxis])
		shearScalar = pt.divide(keypointsTriTipVecX, keypointsTriTipVecY)
		shearMatrix = self.createShearMatrix2Dvec(shearScalar, horizontalAxis=True)
		pixelCoordinates = self.applyShear2D(pixelCoordinates, shearMatrix)
		ATORpt_operations.printPixelCoordinatesIndex(pixelCoordinates, pixelValues, index=0, text="step5")
		#print("5 pixelCoordinates = ", pixelCoordinates)

		posEmbeddingsAbsoluteGeoNormalisedN = pixelCoordinates

		return posEmbeddingsAbsoluteGeoNormalisedN

	def keypointSwap(self, keypoints, keypointAindex, keypointBindex, keypointCindex, axis):
 
		#condition (no swap): keypointA[axis] > keypointB[axis]
		#precondition: number of keypoints = 3
		#precondition: number of geometric dimensions = 2

		keyPointA = keypoints[:, keypointAindex, :]
		keyPointB = keypoints[:, keypointBindex, :]
		keyPointC = keypoints[:, keypointCindex, :]
		#keyPointD = keypoints[:, keypointDindex, :]
		
		keypointAindexNew = pt.gt(keyPointA[:, axis], keyPointB[:, axis])
		keypointBindexNew = pt.logical_not(keypointAindexNew)
		keypointAindexNew = keypointAindexNew.long()
		keypointBindexNew = keypointBindexNew.long()
		keypointAindexNew = keypointAindexNew.unsqueeze(-1)
		keypointBindexNew = keypointBindexNew.unsqueeze(-1)

		keypointsX = keypoints[:, :, xAxis]
		keypointsY = keypoints[:, :, yAxis]
		keyPointAnewX = pt.gather(keypointsX, 1, keypointAindexNew)
		keyPointAnewY = pt.gather(keypointsY, 1, keypointAindexNew)
		keyPointBnewX = pt.gather(keypointsX, 1, keypointBindexNew)
		keyPointBnewY = pt.gather(keypointsY, 1, keypointBindexNew)
		keyPointAnewX = keyPointAnewX.squeeze()
		keyPointAnewY = keyPointAnewY.squeeze()
		keyPointBnewX = keyPointBnewX.squeeze()
		keyPointBnewY = keyPointBnewY.squeeze()
		keyPointAnew = pt.stack([keyPointAnewX, keyPointAnewY], dim=1)
		keyPointBnew = pt.stack([keyPointBnewX, keyPointBnewY], dim=1)

		keypointsNew = keypoints.clone()
		keypointsNew[:, keypointAindex] = keyPointAnew
		keypointsNew[:, keypointBindex] = keyPointBnew
		keypointsNew[:, keypointCindex] = keyPointC

		return keypointsNew

	def applyRotation2D(self, pixelCoordinates, rotationMatrixList):
		if(not useGeometricHashingHardcodedParallelisedRotation):
			xRot = pt.clone(pixelCoordinates)
			for batchIndex, rotationMatrix in enumerate(rotationMatrixList):
				xRotN = pixelCoordinates[batchIndex] @ rotationMatrix.t() # same as x_rot = (rot @ x.t()).t() due to rot in O(n)
				xRot[batchIndex] = xRotN
		else:
			print("useGeometricHashingHardcodedParallelisedRotation incomplete")
		return xRot

	def createRotationMatrix2Dvec(self, vec):
		phi = self.calculateAngleOfVector(vec)
		rotationMatrix = self.createRotationMatrix2D(phi)
		return rotationMatrix

	def createRotationMatrix2D(self, phi):
		#phi = pt.tensor(deg * math.pi / 180)
		s = pt.sin(phi)
		c = pt.cos(phi)
		if(not useGeometricHashingHardcodedParallelisedRotation):
			rotationMatrixList = []
			batchSize = phi.shape[0] 
			for batchIndex in range(batchSize):
				if(xAxis == 0):
					rotationMatrix = pt.tensor([[c[batchIndex], -s[batchIndex]], [s[batchIndex], c[batchIndex]]])	#pt.stack([pt.stack([c[batchIndex], -s[batchIndex]]), pt.stack([s[batchIndex], c[batchIndex]])])
				else:
					rotationMatrix = pt.tensor([[-s[batchIndex], c[batchIndex]], [c[batchIndex], s[batchIndex]]])
				rotationMatrixList.append(rotationMatrix)
		else:
			print("useGeometricHashingHardcodedParallelisedRotation incomplete")
		return rotationMatrixList

	def applyShear2D(self, pixelCoordinates, shearMatrixList):
		if(not useGeometricHashingHardcodedParallelisedRotation):
			xRot = pt.clone(pixelCoordinates)
			for batchIndex, shearMatrix in enumerate(shearMatrixList):	
				xRotN = pixelCoordinates[batchIndex] @ shearMatrix.t() # same as x_rot = (rot @ x.t()).t() due to rot in O(n)
				xRot[batchIndex] = xRotN
		else:
			print("useGeometricHashingHardcodedParallelisedRotation incomplete")
		return xRot

	def createShearMatrix2Dvec(self, shearScalar, horizontalAxis):
		#theta = self.calculateAngleOfVector(vec)  #CHECKTHIS
		#print("theta.shape = ", theta.shape)
		shearMatrix = self.createShearMatrix2D(shearScalar, horizontalAxis)
		return shearMatrix

	def createShearMatrix2D(self, m, horizontalAxis):
		#https://stackoverflow.com/questions/64394325/how-do-i-create-a-shear-matrix-for-pytorchs-f-affine-grid-f-grid-sample
		#m = 1 / pt.tan(pt.tensor(theta))
		#print("m = ", m)
		if(not useGeometricHashingHardcodedParallelisedRotation):
			shearMatrixList = []
			batchSize = m.shape[0] 
			for batchIndex in range(batchSize):
				if((horizontalAxis and xAxis == 0) or (not horizontalAxis and xAxis != 0)):
					shearMatrix = pt.tensor([[1, 0], [m[batchIndex], 1]])	
				else:
					shearMatrix = pt.tensor([[1, m[batchIndex]], [0, 1]])
					
				shearMatrixList.append(shearMatrix)
		else:
			print("useGeometricHashingHardcodedParallelisedRotation incomplete")
		return shearMatrixList
		
	def calculateAngleOfVector(self, vec1):
		#radians
		#calculate angle of vector relative to positive x axis
		batchSize = vec1.shape[0]
		if(xAxis == 0):
			vec2 = pt.unsqueeze(pt.tensor([1.0, 0.0]), 0).repeat(batchSize, 1)
		else:
			vec2 = pt.unsqueeze(pt.tensor([0.0, 1.0]), 0).repeat(batchSize, 1)
		angle = self.calculateAngleBetweenVectors2D(vec1, vec2)
		#angle = pt.angle(vec1)
		return angle
		
	def calculateAngleBetweenVectors2D(self, vec1, vec2):
		#radians
		#if(vect2[xAxis] == vect1[xAxis]):
		#	angleBetweenVectors2D = 0.0
		#else:
		#	angleBetweenVectors2D = pt.atan((vect2[yAxis] - vect1[yAxis]) / (vect2[xAxis] - vect1[xAxis]))
		numerator = self.batchedDotProduct(vec1, vec2)
		denominator = pt.multiply(pt.linalg.norm(vec1, dim=1), pt.linalg.norm(vec2, dim=1)) 
		angleBetweenVectors2D = pt.acos(pt.divide(numerator, denominator))	#interior angle
		return angleBetweenVectors2D;
	
	def batchedDotProduct(self, vec1, vec2):
		#batchedDot = pt.dot(vec1, vec2)
		batchedDot = pt.sum(pt.multiply(vec1, vec2), dim=1)
		return batchedDot
