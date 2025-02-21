"""ATORpt_PTgeometricHashing2DOD.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt PT geometric Hashing 2DOD - parallel processing of ATOR geometric hashing for 2D object data

"""

import torch as pt
import math

from ATORpt_PTglobalDefs import *
import ATORpt_operations

def performGeometricHashingParallel(keypointCoordinates, meshCoordinates, meshValues=None, meshFaces=None):
	#keypointCoordinates shape: [numberPolys, snapshotNumberOfKeypoints, numberOfGeometricDimensions2DOD], i.e. [yCoordinate, xCoordinate] for each pixel in meshCoordinates
	#meshCoordinates shape: [numberPolys, numberCoordinatesInSnapshot, numberOfGeometricDimensions2DOD], i.e. [yCoordinate, xCoordinate] for each keypoint in poly
	use3DOD = False
	
	if(debugVerbose):
		print("start performGeometricHashingParallel")

	#based on https://patentscope.wipo.int/search/en/detail.jsf?docId=WO2011088497 Fig 30->35
	#see ATORmethod2DOD:transformObjectData2DOD for unvectorised (ie serial processing) method
	
	#transformedKeypointCoordinates should equal transformedKeypointCoordinatesExpected (transformedObjectTriangleCoordinatesExpected) at end of transformation
	transformedKeypointCoordinatesExpected = pt.zeros([3, 2]).to(device)
	transformedKeypointCoordinatesExpected[0][xAxisGeometricHashing] = normalisedObjectTriangleBaseLength/2
	transformedKeypointCoordinatesExpected[0][yAxisGeometricHashing] = 0.0
	transformedKeypointCoordinatesExpected[1][xAxisGeometricHashing] = -normalisedObjectTriangleBaseLength/2
	transformedKeypointCoordinatesExpected[1][yAxisGeometricHashing] = 0.0
	transformedKeypointCoordinatesExpected[2][xAxisGeometricHashing] = 0
	transformedKeypointCoordinatesExpected[2][yAxisGeometricHashing] = normalisedObjectTriangleHeight
	
	transformedMeshCoordinates = meshCoordinates
	transformedKeypointCoordinates = keypointCoordinates	#retain original for calculations	#.copy()
	ATORpt_operations.printCoordinatesIndex(use3DOD, transformedKeypointCoordinates, transformedMeshCoordinates, meshValues, meshFaces, index=debugPolyIndex, step=0)	#will be out of range of render view port
	
	#ATORpt_operations.printCoordinates(use3DOD, transformedKeypointCoordinates, transformedMeshCoordinates, meshValues, meshFaces, step=0, centreSnapshots=False)
	
	#keypointCoordinates;
	# kp2
	#  \\
	#   \_\
	#  kp1 kp0  
	
	#apply hardcoded geometric hashing function;

	#step 1 (shift x/y - wrt centre of keypointCoordinates [0, 1]):
	#translate object data on X and Y axis such that the object triangle base is positioned at centre of keypointCoordinates [0, 1]):
	#Fig 31
	keypointsTriBaseCentre = pt.add(keypointCoordinates[:, 0], keypointCoordinates[:, 1])/2.0
	transformedMeshCoordinates = pt.subtract(transformedMeshCoordinates, keypointsTriBaseCentre.unsqueeze(1))
	transformedKeypointCoordinates = pt.subtract(transformedKeypointCoordinates, keypointsTriBaseCentre.unsqueeze(1))
	ATORpt_operations.printCoordinatesIndex(use3DOD, transformedKeypointCoordinates, transformedMeshCoordinates, meshValues, meshFaces, index=debugPolyIndex, step=1)
	
	#step 2 (rotate - wrt keypointCoordinates [0, 1]):
	#2ia. rotate object data such that the object triangle side is parallel with X axis [and 2ii. third apex is above the lowest 2 apexes]
	#Fig 31
	#print("keypointCoordinates[:, 1] = ", keypointCoordinates[:, 1])
	#print("keypointCoordinates[:, 0] = ", keypointCoordinates[:, 0])
	keypointsTriBaseVec = -pt.subtract(transformedKeypointCoordinates[:, 1], transformedKeypointCoordinates[:, 0])
	rotationMatrix = createRotationMatrix2Dvec(keypointsTriBaseVec)
	transformedMeshCoordinates = applyRotation2D(transformedMeshCoordinates, rotationMatrix)
	transformedKeypointCoordinates = applyRotation2D(transformedKeypointCoordinates, rotationMatrix)
	ATORpt_operations.printCoordinatesIndex(use3DOD, transformedKeypointCoordinates, transformedMeshCoordinates, meshValues, meshFaces, index=debugPolyIndex, step=2)
	
	#step 3 (scale x - wrt object triangle [0, 1]):   
	#1a. Scale object data such that the object triangle side is of same length as a predefined side of a predefined triangle
	#Fig 33
	keypointsTriBaseSizeX = calculateDistance(transformedKeypointCoordinates[:, 0], transformedKeypointCoordinates[:, 1]) / normalisedObjectTriangleBaseLength	#or calculateDistance(transformedMeshCoordinates[:, 0], transformedMeshCoordinates[:, 1])
	transformedMeshCoordinates[:, :, xAxisGeometricHashing] = pt.divide(transformedMeshCoordinates[:, :, xAxisGeometricHashing] , keypointsTriBaseSizeX.unsqueeze(1))
	transformedKeypointCoordinates[:, :, xAxisGeometricHashing] = pt.divide(transformedKeypointCoordinates[:, :, xAxisGeometricHashing] , keypointsTriBaseSizeX.unsqueeze(1))
	ATORpt_operations.printCoordinatesIndex(use3DOD, transformedKeypointCoordinates, transformedMeshCoordinates, meshValues, meshFaces, index=debugPolyIndex, step=3)

	#step 4 (scale y - wrt object triangle [1y, 2y]):
	#3a. Scale object data on Y axis such that the third apex is the same perpendicular distance away from the side as is the case for the predefined triangle.
	#Fig 34
	transformedKeypointsTriBaseCentre = pt.add(transformedKeypointCoordinates[:, 0], transformedKeypointCoordinates[:, 1])/2.0
	keypointsTriHeightSize = calculateDistance(transformedKeypointCoordinates[:, 2], transformedKeypointsTriBaseCentre) / normalisedObjectTriangleHeight	#or calculateDistance(transformedMeshCoordinates[:, 2], keypointsTriBaseCentre
	transformedMeshCoordinates[:, :, yAxisGeometricHashing] = pt.divide(transformedMeshCoordinates[:, :, yAxisGeometricHashing], keypointsTriHeightSize.unsqueeze(1))
	transformedKeypointCoordinates[:, :, yAxisGeometricHashing] = pt.divide(transformedKeypointCoordinates[:, :, yAxisGeometricHashing], keypointsTriHeightSize.unsqueeze(1))
	ATORpt_operations.printCoordinatesIndex(use3DOD, transformedKeypointCoordinates, transformedMeshCoordinates, meshValues, meshFaces, index=debugPolyIndex, step=4)
	
	#step 5 (shear):
	#4a. shear object data along X axis such that object triangle apexes are coincident with predefined triangle apexes
	#Fig 35
	shearScalar = -transformedKeypointCoordinates[:, 2, xAxisGeometricHashing]/normalisedObjectTriangleHeight
	shearMatrix = createShearMatrix2Dvec(shearScalar, horizontalAxis=True)
	transformedMeshCoordinates = applyShear2D(transformedMeshCoordinates, shearMatrix)
	transformedKeypointCoordinates = applyShear2D(transformedKeypointCoordinates, shearMatrix)
	ATORpt_operations.printCoordinatesIndex(use3DOD, transformedKeypointCoordinates, transformedMeshCoordinates, meshValues, meshFaces, index=debugPolyIndex, step=5)
	
	ATORpt_operations.printCoordinates(use3DOD, transformedKeypointCoordinates, transformedMeshCoordinates, meshValues, meshFaces, step=5, centreSnapshots=False)

	#print("transformedKeypointCoordinatesExpected = ", transformedKeypointCoordinatesExpected)
	
	return transformedMeshCoordinates


def createRotationMatrix2Dvec(vec):
	phi = calculateAngleOfVector(vec)
	rotationMatrix = createRotationMatrix2D(phi)
	return rotationMatrix

def createRotationMatrix2D(phi):
	s = pt.sin(phi)
	c = pt.cos(phi)
	if(useGeometricHashingHardcodedParallelisedDeformation):
		rotationMatrix = pt.stack([pt.stack([c, -s], dim=-1), pt.stack([s, c], dim=-1)], dim=-2)
	else:
		rotationMatrixList = []
		batchSize = phi.shape[0] 
		for batchIndex in range(batchSize):
			if(xAxisGeometricHashing == 0):
				rotationMatrix = pt.tensor([[c[batchIndex], -s[batchIndex]], [s[batchIndex], c[batchIndex]]])	#pt.stack([pt.stack([c[batchIndex], -s[batchIndex]]), pt.stack([s[batchIndex], c[batchIndex]])])
			else:
				rotationMatrix = pt.tensor([[-s[batchIndex], c[batchIndex]], [c[batchIndex], s[batchIndex]]])
			rotationMatrixList.append(rotationMatrix)
		rotationMatrix = rotationMatrixList
	return rotationMatrix

def applyRotation2D(meshCoordinates, rotationMatrix):
	if(useGeometricHashingHardcodedParallelisedDeformation):
		xRot = pt.bmm(meshCoordinates, rotationMatrix)	#.transpose(1, 2)
	else:
		rotationMatrixList = rotationMatrix
		xRot = pt.clone(meshCoordinates)
		for batchIndex, rotationMatrix in enumerate(rotationMatrixList):
			xRotN = meshCoordinates[batchIndex] @ rotationMatrix.t() # same as x_rot = (rot @ x.t()).t() due to rot in O(n)
			xRot[batchIndex] = xRotN
	return xRot

def createShearMatrix2Dvec(shearScalar, horizontalAxis):
	shearMatrix = createShearMatrix2D(shearScalar, horizontalAxis)
	return shearMatrix

def createShearMatrix2D(m, horizontalAxis):
	#https://stackoverflow.com/questions/64394325/how-do-i-create-a-shear-matrix-for-pytorchs-f-affine-grid-f-grid-sample
	#m = 1 / pt.tan(pt.tensor(theta))
	if(useGeometricHashingHardcodedParallelisedDeformation):
		b = m.shape[0]
		if((horizontalAxis and xAxisGeometricHashing == 0) or (not horizontalAxis and xAxisGeometricHashing != 0)):
			shearMatrix = pt.stack([pt.stack([pt.ones(b), pt.zeros(b)], dim=-1), pt.stack([m, pt.ones(b)], dim=-1)], dim=-2)
		else:
			shearMatrix = pt.stack([pt.stack([pt.ones(b), m], dim=-1), pt.stack([pt.zeros(b), pt.ones(b)], dim=-1)], dim=-2)
	else:
		shearMatrixList = []
		batchSize = m.shape[0] 
		for batchIndex in range(batchSize):
			if((horizontalAxis and xAxisGeometricHashing == 0) or (not horizontalAxis and xAxisGeometricHashing != 0)):
				shearMatrix = pt.tensor([[1, 0], [m[batchIndex], 1]])	
			else:
				shearMatrix = pt.tensor([[1, m[batchIndex]], [0, 1]])
			shearMatrixList.append(shearMatrix)
		shearMatrix = shearMatrixList
	return shearMatrix

def applyShear2D(meshCoordinates, shearMatrix):
	if(useGeometricHashingHardcodedParallelisedDeformation):
		xRot = pt.bmm(meshCoordinates, shearMatrix)	#.transpose(1, 2)
	else:
		shearMatrixList = shearMatrix
		xRot = pt.clone(meshCoordinates)
		for batchIndex, shearMatrix in enumerate(shearMatrixList):	
			xRotN = meshCoordinates[batchIndex] @ shearMatrix.t() # same as x_rot = (rot @ x.t()).t() due to rot in O(n)
			xRot[batchIndex] = xRotN
	return xRot

def calculateAngleOfVector(vec1):
	#radians
	angle = pt.atan2(vec1[:, 1], vec1[:, 0])
	return angle

def calculateAngleBetweenVectors2D(vec1, vec2):
	numerator = batchedDotProduct(vec1, vec2)
	denominator = pt.multiply(pt.linalg.norm(vec1, dim=1), pt.linalg.norm(vec2, dim=1)) 
	angleBetweenVectors2D = pt.acos(pt.divide(numerator, denominator))	#interior angle
	return angleBetweenVectors2D;

def batchedDotProduct(vec1, vec2):
	batchedDot = pt.sum(pt.multiply(vec1, vec2), dim=1)
	return batchedDot

def calculateDistance(keypoint1, keypoint2):
	#dist = sqrt((x1-x2)^2 + (y1-y2)^2) 
	xDiff = keypoint1[:, xAxisGeometricHashing] - keypoint2[:, xAxisGeometricHashing] 
	yDiff = keypoint1[:, yAxisGeometricHashing] - keypoint2[:, yAxisGeometricHashing] 
	distance = pt.sqrt(pt.add(pt.square(xDiff), pt.square(yDiff))) 
	return distance
	
