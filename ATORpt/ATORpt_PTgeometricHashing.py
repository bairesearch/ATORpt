"""ATORpt_PTgeometricHashing.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt PT geometric Hashing - parallel processing of ATOR geometric hashing

"""

import torch as pt

from ATORpt_globalDefs import *
import ATORpt_operations

def performGeometricHashingParallel(keypointCoordinates, pixelCoordinates, pixelValues=None):
	#pixelCoordinates shape: [numberCoordinates, numberOfGeometricDimensions], i.e. [yCoordinate, xCoordinate] for each pixel in pixelCoordinates
	#debugGeometricHashingParallel requires pixelValues and pixelCoordinates=snapshotPixelCoordinates (not snapshotPixelCoordinates)
	
	print("start performGeometricHashingParallel")

	#based on https://patentscope.wipo.int/search/en/detail.jsf?docId=WO2011088497 Fig 30->35
	#see ATORmethod2DOD:transformObjectData2DOD for unvectorised method

	#implementation limitation: assume input image is roughly upright; only perform 1 geometric hashing transformation (not geometricHashingNumKeypoints transformations, based on every possible permutation of keypoints)
	
	if(debugGeometricHashingParallel):
		#artificially set position of first pixel and first set of keypoints to good combination for visualisation
		firstSequenceIndex = 0
		pixelCoordinates[firstSequenceIndex][:][xAxis] = 0.2
		pixelCoordinates[firstSequenceIndex][:][yAxis] = 0.6
		#pixelValues[firstSequenceIndex] = 1.0
		keypointCoordinates[firstSequenceIndex][0][xAxis] = 0.7
		keypointCoordinates[firstSequenceIndex][0][yAxis] = 0.3
		keypointCoordinates[firstSequenceIndex][1][xAxis] = 0.4
		keypointCoordinates[firstSequenceIndex][1][yAxis] = 0.2
		keypointCoordinates[firstSequenceIndex][2][xAxis] = 0.1
		keypointCoordinates[firstSequenceIndex][2][yAxis] = 0.8
	
	#print("pixelValues[0] = ", pixelValues[0])
	print("pixelCoordinates.shape = ", pixelCoordinates.shape)
	print("keypointCoordinates.shape = ", keypointCoordinates.shape)
	
	#ATORpt_operations.printPixelCoordinates(pixelCoordinates, pixelValues)
	#ATORpt_operations.printKeypoints(keypointCoordinates)
	#ATORpt_operations.printPixelCoordinatesIndex(pixelCoordinates, pixelValues, index=0, text="step0")
	ATORpt_operations.printKeypointsIndex(keypointCoordinates, index=0)
	#print("0 pixelCoordinates = ", pixelCoordinates)
	

	#reorder keypointCoordinates;
	# kp2
	#  \\
	#   \_\
	#  kp1 kp0  

	keypointCoordinates = keypointSwap(keypointCoordinates, 2, 1, 0, yAxis)
	keypointCoordinates = keypointSwap(keypointCoordinates, 2, 0, 1, yAxis)
	keypointCoordinates = keypointSwap(keypointCoordinates, 0, 1, 2, xAxis)

	#apply hardcoded geometric hashing function;

	#step 1 (shift x/y - wrt centre of keypointCoordinates [0, 1]):
	#translate object data on X and Y axis such that the object triangle base is positioned at centre of keypointCoordinates [0, 1]):
	#Fig 31
	keypointsTriBaseCentre = pt.add(keypointCoordinates[:, 0], keypointCoordinates[:, 1])/2.0
	pixelCoordinates = pt.subtract(pixelCoordinates, keypointsTriBaseCentre.unsqueeze(1))
	ATORpt_operations.printPixelCoordinatesIndex(pixelCoordinates, pixelValues, index=0, text="step1")
	#print("1 pixelCoordinates = ", pixelCoordinates)

	#step 2 (rotate - wrt keypointCoordinates [0, 1]):
	#2ia. rotate object data such that the object triangle side is parallel with X axis [and 2ii. third apex is above the lowest 2 apexes]
	#Fig 31
	keypointsTriBaseVec = pt.subtract(keypointCoordinates[:, 1], keypointCoordinates[:, 0])
	rotationMatrix = createRotationMatrix2Dvec(keypointsTriBaseVec)
	print("pixelCoordinates.shape = ", pixelCoordinates.shape)
	print("rotationMatrix.shape = ", rotationMatrix.shape)
	pixelCoordinates = applyRotation2D(pixelCoordinates, rotationMatrix)
	ATORpt_operations.printPixelCoordinatesIndex(pixelCoordinates, pixelValues, index=0, text="step2")
	#print("2 pixelCoordinates = ", pixelCoordinates)

	#step 3 (scale x - wrt keypointCoordinates [0, 1]):   
	#1a. Scale object data such that the object triangle side is of same length as a predefined side of a predefined triangle
	#Fig 33
	keypointsTriBaseSizeX = pt.subtract(keypointCoordinates[:, 0, xAxis], keypointCoordinates[:, 1, xAxis])
	pixelsX = pixelCoordinates[:, :, xAxis] 
	pixelsX = pt.divide(pixelsX, keypointsTriBaseSizeX.unsqueeze(1))
	pixelCoordinates[:, :, xAxis] = pixelsX
	ATORpt_operations.printPixelCoordinatesIndex(pixelCoordinates, pixelValues, index=0, text="step3")
	#print("3 pixelCoordinates = ", pixelCoordinates)

	#step 4 (scale y - wrt keypointCoordinates [1y, 2y]):
	#3a. Scale object data on Y axis such that the third apex is the same perpendicular distance away from the side as is the case for the predefined triangle.
	#Fig 34
	keypointsTriHeightSize = pt.subtract(keypointCoordinates[:, 2, yAxis], keypointCoordinates[:, 1, yAxis])
	pixelsY = pixelCoordinates[:, :, yAxis]
	pixelsY = pt.divide(pixelsY, keypointsTriHeightSize.unsqueeze(1))
	pixelCoordinates[:, :, yAxis] = pixelsY
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
	shearMatrix = createShearMatrix2Dvec(shearScalar, horizontalAxis=True)
	pixelCoordinates = applyShear2D(pixelCoordinates, shearMatrix)
	ATORpt_operations.printPixelCoordinatesIndex(pixelCoordinates, pixelValues, index=0, text="step5")
	#print("5 pixelCoordinates = ", pixelCoordinates)

	transformedPixelCoordinates = pixelCoordinates
	return transformedPixelCoordinates

def keypointSwap(keypoints, keypointAindex, keypointBindex, keypointCindex, axis):

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

def createRotationMatrix2Dvec(vec):
	phi = calculateAngleOfVector(vec)
	print("phi.shape = ", phi.shape)
	rotationMatrix = createRotationMatrix2D(phi)
	return rotationMatrix

def createRotationMatrix2D(phi):
	#phi = pt.tensor(deg * math.pi / 180)
	s = pt.sin(phi)
	c = pt.cos(phi)
	print("s.shape = ", s.shape)
	if(useGeometricHashingHardcodedParallelisedDeformation):
		rotationMatrix = pt.stack([pt.stack([c, -s], dim=-1), pt.stack([s, c], dim=-1)], dim=-2)
		print("1 rotationMatrix.shape = ", rotationMatrix.shape)
	else:
		rotationMatrixList = []
		batchSize = phi.shape[0] 
		for batchIndex in range(batchSize):
			if(xAxis == 0):
				rotationMatrix = pt.tensor([[c[batchIndex], -s[batchIndex]], [s[batchIndex], c[batchIndex]]])	#pt.stack([pt.stack([c[batchIndex], -s[batchIndex]]), pt.stack([s[batchIndex], c[batchIndex]])])
			else:
				rotationMatrix = pt.tensor([[-s[batchIndex], c[batchIndex]], [c[batchIndex], s[batchIndex]]])
			rotationMatrixList.append(rotationMatrix)
		rotationMatrix = rotationMatrixList
	return rotationMatrix

def applyRotation2D(pixelCoordinates, rotationMatrix):
	if(useGeometricHashingHardcodedParallelisedDeformation):
		print("applyRotation2D: 1 pixelCoordinates.shape = ", pixelCoordinates.shape)
		print("applyRotation2D: 1 rotationMatrix.shape = ", rotationMatrix.shape)
		xRot = pt.bmm(pixelCoordinates, rotationMatrix.transpose(1, 2))
		print("applyRotation2D: 2 pixelCoordinates.shape = ", pixelCoordinates.shape)
	else:
		rotationMatrixList = rotationMatrix
		xRot = pt.clone(pixelCoordinates)
		for batchIndex, rotationMatrix in enumerate(rotationMatrixList):
			xRotN = pixelCoordinates[batchIndex] @ rotationMatrix.t() # same as x_rot = (rot @ x.t()).t() due to rot in O(n)
			xRot[batchIndex] = xRotN
	return xRot

def createShearMatrix2Dvec(shearScalar, horizontalAxis):
	#theta = calculateAngleOfVector(vec)  #CHECKTHIS
	print("shearScalar.shape = ", shearScalar.shape)
	shearMatrix = createShearMatrix2D(shearScalar, horizontalAxis)
	return shearMatrix

def createShearMatrix2D(m, horizontalAxis):
	#https://stackoverflow.com/questions/64394325/how-do-i-create-a-shear-matrix-for-pytorchs-f-affine-grid-f-grid-sample
	#m = 1 / pt.tan(pt.tensor(theta))
	if(useGeometricHashingHardcodedParallelisedDeformation):
		b = m.shape[0]
		if((horizontalAxis and xAxis == 0) or (not horizontalAxis and xAxis != 0)):
			shearMatrix = pt.stack([pt.stack([pt.ones(b), pt.zeros(b)], dim=-1), pt.stack([m, pt.ones(b)], dim=-1)], dim=-2)
		else:
			shearMatrix = pt.stack([pt.stack([pt.ones(b), m], dim=-1), pt.stack([pt.zeros(b), pt.ones(b)], dim=-1)], dim=-2)
		print("1 shearMatrix.shape = ", shearMatrix.shape)
	else:
		shearMatrixList = []
		batchSize = m.shape[0] 
		for batchIndex in range(batchSize):
			if((horizontalAxis and xAxis == 0) or (not horizontalAxis and xAxis != 0)):
				shearMatrix = pt.tensor([[1, 0], [m[batchIndex], 1]])	
			else:
				shearMatrix = pt.tensor([[1, m[batchIndex]], [0, 1]])
			shearMatrixList.append(shearMatrix)
		shearMatrix = shearMatrixList
	return shearMatrix

def applyShear2D(pixelCoordinates, shearMatrix):
	if(useGeometricHashingHardcodedParallelisedDeformation):
		print("applyShear2D: 1 pixelCoordinates.shape = ", pixelCoordinates.shape)
		xRot = pt.bmm(pixelCoordinates, shearMatrix.transpose(1, 2))
		print("applyShear2D: 2 pixelCoordinates.shape = ", pixelCoordinates.shape)
	else:
		shearMatrixList = shearMatrix
		xRot = pt.clone(pixelCoordinates)
		for batchIndex, shearMatrix in enumerate(shearMatrixList):	
			xRotN = pixelCoordinates[batchIndex] @ shearMatrix.t() # same as x_rot = (rot @ x.t()).t() due to rot in O(n)
			xRot[batchIndex] = xRotN
	return xRot

def calculateAngleOfVector(vec1):
	#radians
	#calculate angle of vector relative to positive x axis
	batchSize = vec1.shape[0]
	if(xAxis == 0):
		vec2 = pt.unsqueeze(pt.tensor([1.0, 0.0]), 0).repeat(batchSize, 1)
	else:
		vec2 = pt.unsqueeze(pt.tensor([0.0, 1.0]), 0).repeat(batchSize, 1)
	angle = calculateAngleBetweenVectors2D(vec1, vec2)
	#angle = pt.angle(vec1)
	return angle

def calculateAngleBetweenVectors2D(vec1, vec2):
	#radians
	#if(vect2[xAxis] == vect1[xAxis]):
	#	angleBetweenVectors2D = 0.0
	#else:
	#	angleBetweenVectors2D = pt.atan((vect2[yAxis] - vect1[yAxis]) / (vect2[xAxis] - vect1[xAxis]))
	numerator = batchedDotProduct(vec1, vec2)
	denominator = pt.multiply(pt.linalg.norm(vec1, dim=1), pt.linalg.norm(vec2, dim=1)) 
	angleBetweenVectors2D = pt.acos(pt.divide(numerator, denominator))	#interior angle
	return angleBetweenVectors2D;

def batchedDotProduct(vec1, vec2):
	#batchedDot = pt.dot(vec1, vec2)
	batchedDot = pt.sum(pt.multiply(vec1, vec2), dim=1)
	return batchedDot

