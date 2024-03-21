"""ATORpt_PTkeypoints.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt PT keypoints

"""

import torch as pt
import numpy as np
import torchvision.transforms.functional as TF

from ATORpt_globalDefs import *
import ATORpt_operations

if(fullRotationalInvariance):
	def generateKeypointPermutations(keypointCoordinates):			
		#make kp2 the primary keypoint of the poly rotational permutation (kp0, kp1 will be swapped later if necessary to ensure clockwise definition of keypoints within poly tri)
		keypointCoordinatesPerm1 = keypointCoordinates
		keypointCoordinatesPerm2 = keypointSwap(keypointCoordinates, 2, 0)
		keypointCoordinatesPerm3 = keypointSwap(keypointCoordinates, 2, 1)
		keypointCoordinates = pt.cat((keypointCoordinatesPerm1, keypointCoordinatesPerm2, keypointCoordinatesPerm3), dim=0)		
		return keypointCoordinates

	def keypointSwap(keypoints, keypointAindex, keypointBindex):
		keyPointANew = keypoints[:, keypointBindex].clone()
		keyPointBNew = keypoints[:, keypointAindex].clone()
		keypoints[:, keypointAindex] = keyPointANew
		keypoints[:, keypointBindex] = keyPointBNew
		#print("keypointSwap: out keypoints = ", keypoints)
		return keypoints
		
	def reorderKeypoints(keypointCoordinates):

		#reorder keypointCoordinates (ensure clockwise definition of keypoints within poly tri);
		# kp2
		#  \\
		#   \_\
		#  kp1 kp0  

		keypointCoordinates = keypointClockwiseOrder(keypointCoordinates)

		return keypointCoordinates
		
else:
	def reorderKeypoints(keypointCoordinates):

		#reorder keypointCoordinates (ensure clockwise definition of keypoints within poly tri, and kp2.y is highest of all);
		# kp2
		#  \\
		#   \_\
		#  kp1 kp0  

		keypointCoordinates = keypointClockwiseOrder(keypointCoordinates)
		keypointCoordinates = keypointConditionalRotate(keypointCoordinates, 2, 0, 1, yAxisGeometricHashing)
		keypointCoordinates = keypointConditionalRotate(keypointCoordinates, 2, 0, 1, yAxisGeometricHashing)

		'''
		keypointCoordinates = keypointConditionalSwap(keypointCoordinates, 2, 1, 1, yAxisGeometricHashing)
		keypointCoordinates = keypointConditionalSwap(keypointCoordinates, 2, 0, 0, yAxisGeometricHashing)
		keypointCoordinates = keypointConditionalSwap(keypointCoordinates, 0, 1, 1, xAxisGeometricHashing)
		'''
		
		return keypointCoordinates

	def keypointConditionalRotate(keypoints, keypointAindex, keypointBindex, keypointCindex, axis):
	
		#condition (no swap): keypointA[axis] > max(keypointB[axis], keypointC[axis]) (element wise test)
		#else rotate keypoints clockwise (element wise)
		#precondition: number of keypoints = 3
		#precondition: number of geometric dimensions = 2
				
		keyPointA = keypoints[:, keypointAindex]
		keyPointB = keypoints[:, keypointBindex]
		keyPointC = keypoints[:, keypointCindex]
		keyPointBCaxis = pt.stack((keyPointB[:, axis], keyPointC[:, axis]), dim=0)
		keyPointMaxBCaxis = pt.max(keyPointBCaxis, dim=0).values

		keyPointANewX = pt.where(keyPointA[:, axis] > keyPointMaxBCaxis, keyPointA[:, xAxisGeometricHashing], keyPointC[:, xAxisGeometricHashing])
		keyPointANewY = pt.where(keyPointA[:, axis] > keyPointMaxBCaxis, keyPointA[:, yAxisGeometricHashing], keyPointC[:, yAxisGeometricHashing])
		keyPointBNewX = pt.where(keyPointA[:, axis] > keyPointMaxBCaxis, keyPointB[:, xAxisGeometricHashing], keyPointA[:, xAxisGeometricHashing])
		keyPointBNewY = pt.where(keyPointA[:, axis] > keyPointMaxBCaxis, keyPointB[:, yAxisGeometricHashing], keyPointA[:, yAxisGeometricHashing])
		keyPointCNewX = pt.where(keyPointA[:, axis] > keyPointMaxBCaxis, keyPointC[:, xAxisGeometricHashing], keyPointB[:, xAxisGeometricHashing])
		keyPointCNewY = pt.where(keyPointA[:, axis] > keyPointMaxBCaxis, keyPointC[:, yAxisGeometricHashing], keyPointB[:, yAxisGeometricHashing])
		keyPointANew = pt.stack([keyPointANewX, keyPointANewY], dim=1)
		keyPointBNew = pt.stack([keyPointBNewX, keyPointBNewY], dim=1)
		keyPointCNew = pt.stack([keyPointCNewX, keyPointCNewY], dim=1)
		keypoints[:, keypointAindex] = keyPointANew
		keypoints[:, keypointBindex] = keyPointBNew
		keypoints[:, keypointCindex] = keyPointCNew
		
		if(debugGeometricHashingParallel):	
			print("keypointConditionalRotate(keypointCoordinates, keypointAindex=", keypointAindex, ", keypointBindex=",  keypointBindex, ", keypointCindex=", keypointCindex, ", axis=", axis)
			print("keypoints[0] = ", keypoints[0])

		return keypoints
		
	'''
	def keypointConditionalSwap(keypoints, keypointAindex, keypointBindex, keypointCindex, axis):

		#condition (no swap): keypointA[axis] > keypointB[axis] (element wise test)
		#else swap keypointA for keypointC (element wise)
		#precondition: number of keypoints = 3
		#precondition: number of geometric dimensions = 2

		keyPointA = keypoints[:, keypointAindex]
		keyPointB = keypoints[:, keypointBindex]
		keyPointC = keypoints[:, keypointCindex]

		keyPointANewX = pt.where(keyPointA[:, axis] > keyPointB[:, axis], keyPointA[:, xAxisGeometricHashing], keyPointC[:, xAxisGeometricHashing])
		keyPointANewY = pt.where(keyPointA[:, axis] > keyPointB[:, axis], keyPointA[:, yAxisGeometricHashing], keyPointC[:, yAxisGeometricHashing])
		keyPointCNewX = pt.where(keyPointA[:, axis] > keyPointB[:, axis], keyPointC[:, xAxisGeometricHashing], keyPointA[:, xAxisGeometricHashing])
		keyPointCNewY = pt.where(keyPointA[:, axis] > keyPointB[:, axis], keyPointC[:, yAxisGeometricHashing], keyPointA[:, yAxisGeometricHashing])
		keyPointANew = pt.stack([keyPointANewX, keyPointANewY], dim=1)
		keyPointCNew = pt.stack([keyPointCNewX, keyPointCNewY], dim=1)
		keypoints[:, keypointAindex] = keyPointANew
		keypoints[:, keypointCindex] = keyPointCNew

		if(debugGeometricHashingParallel):	
			print("keypointConditionalSwap(keypointCoordinates, keypointAindex=", keypointAindex, ", keypointBindex=",  keypointBindex, ", keypointCindex=", keypointCindex, ", axis=", axis)
			print("keypoints[0] = ", keypoints[0])

		return keypoints
	'''

def keypointClockwiseOrder(keypoints):
	#make kp2 the primary keypoint of the poly rotational permutation (kp0, kp1 will be swapped if necessary to ensure clockwise definition of keypoints within poly tri)
	signedTriArea = -calculateSignedArea(keypoints)	#CHECKTHIS: -
	#print("keypoints = ", keypoints)
	#print("signedTriArea = ", signedTriArea)
	keyPointA = keypoints[:, keypointAindex]
	keyPointB = keypoints[:, keypointBindex]
	keyPointC = keypoints[:, keypointCindex]
	#Swap A and B to ensure clockwise order
	keyPointANewX = pt.where(signedTriArea < 0, keyPointB[:, xAxisGeometricHashing], keyPointA[:, xAxisGeometricHashing])
	keyPointANewY = pt.where(signedTriArea < 0, keyPointB[:, yAxisGeometricHashing], keyPointA[:, yAxisGeometricHashing])
	keyPointBNewX = pt.where(signedTriArea < 0, keyPointA[:, xAxisGeometricHashing], keyPointB[:, xAxisGeometricHashing])
	keyPointBNewY = pt.where(signedTriArea < 0, keyPointA[:, yAxisGeometricHashing], keyPointB[:, yAxisGeometricHashing])
	keyPointANew = pt.stack([keyPointANewX, keyPointANewY], dim=1)
	keyPointBNew = pt.stack([keyPointBNewX, keyPointBNewY], dim=1)
	keyPointCNew = keyPointC
	keypoints[:, keypointAindex] = keyPointANew
	keypoints[:, keypointBindex] = keyPointBNew
	keypoints[:, keypointCindex] = keyPointCNew
	return keypoints

def calculateSignedArea(keypoints):
	BmAx = (keypoints[:, keypointBindex, xAxisGeometricHashing] - keypoints[:, keypointAindex, xAxisGeometricHashing])
	CmAy = (keypoints[:, keypointCindex, yAxisGeometricHashing] - keypoints[:, keypointAindex, yAxisGeometricHashing])
	CmAx = (keypoints[:, keypointCindex, xAxisGeometricHashing] - keypoints[:, keypointAindex, xAxisGeometricHashing])
	BmAy = (keypoints[:, keypointBindex, yAxisGeometricHashing] - keypoints[:, keypointAindex, yAxisGeometricHashing])
	signedTriArea = ((BmAx * CmAy) - (CmAx * BmAy))
	#dim = [numberOfPolys]
	return signedTriArea

'''
def keypointOrderFlip(keypointCoordinates, meshCoordinates):
	#y axis will not work when kp2 point occurs midway along axis y
	ABy = pt.stack((keypointCoordinates[:, keypointAindex, yAxisGeometricHashing], keypointCoordinates[:, keypointBindex, yAxisGeometricHashing]), dim=0)
	ABmaxY = pt.max(ABy, dim=0).values
	maskFlipY = (keypointCoordinates[:, keypointCindex, yAxisGeometricHashing] < ABmaxY)
	maskFlipX = (keypointCoordinates[:, keypointAindex, xAxisGeometricHashing] < keypointCoordinates[:, keypointBindex, xAxisGeometricHashing])
	keypointCoordinates = flipMatrixAlongDimUsingMask(keypointCoordinates, maskFlipY, 1, yAxisGeometricHashing)
	meshCoordinates = flipMatrixAlongDimUsingMask(meshCoordinates, maskFlipX, 1, xAxisGeometricHashing)
	return keypointCoordinates, meshCoordinates

def flipMatrixAlongDimUsingMask(matrix, mask, dim, axis):
	numPoints = matrix.shape[1]
	maskExtended = mask.unsqueeze(1).repeat(1, numPoints)
	matrixAxis = matrix[:, :, axis]
	matrixAxisFlipped = pt.flip(matrixAxis, dims=[dim])
	matrixAxisNew = pt.where(maskExtended, matrixAxisFlipped, matrixAxis)
	matrix[:, :, axis] = matrixAxisNew
	return matrix
'''
	
	
def padCoordinatesArrays(imageKeypointCoordinates, snapshotPixelCoordinates, snapshotMeshCoordinates, snapshotMeshValues, snapshotMeshFaces, snapshotMeshPolyCoordinates):
	#ensure num of polys is equal per image
	imageKeypointCoordinates = padCoordinatesArray(imageKeypointCoordinates)
	snapshotPixelCoordinates = padCoordinatesArray(snapshotPixelCoordinates)
	snapshotMeshCoordinates = padCoordinatesArray(snapshotMeshCoordinates)
	snapshotMeshValues = padCoordinatesArray(snapshotMeshValues)
	snapshotMeshFaces = padCoordinatesArray(snapshotMeshFaces)
	snapshotMeshPolyCoordinates = padCoordinatesArray(snapshotMeshPolyCoordinates)
	return imageKeypointCoordinates, snapshotPixelCoordinates, snapshotMeshCoordinates, snapshotMeshValues, snapshotMeshFaces, snapshotMeshPolyCoordinates
	
def padCoordinatesArray(coordinates):
	coordinatesPadded = padAlongDimension(coordinates, 0, ATORmaxNumberOfPolys)
	return coordinatesPadded

def padAlongDimension(x, dimToPad, targetSize):
	currentSize = x.shape[0]
	currentType = x.dtype
	padSize = targetSize-currentSize
	if(padSize > 0):
		padShape = list(x.shape)
		padShape[dimToPad] = padSize
		pad = pt.ones(padShape, dtype=currentType).to(device) * meshPadValue
		xPadded = pt.cat((x, pad), dim=dimToPad)
	else:
		xPadded = x
	return xPadded

def cropCoordinatesArray(coordinates, maxNumberOfPolys):
	dimToCrop = 0
	if(coordinates.shape[0] > maxNumberOfPolys):
		coordinates = coordinates[0:maxNumberOfPolys]
	return coordinates
	
def performKeypointDetection(imageFeatureCoordinates):
	#featureCoordinatesList size = batchSize list of [featureIndex, x/yIndex]
	keypointCoordinates = performKeypointDetectionBasic(imageFeatureCoordinates)
	return keypointCoordinates
			
def performKeypointDetectionBasic(featureCoordinates):

	if(keypointDetectionCriteria):
		#criteria 0: ensure keypoints are not same
		maskNotSame = keypointDetectionNotSame(featureCoordinates)
		featureCoordinates = featureCoordinates[maskNotSame]
	
	#based on ATORpt_RFapply:generateRFtypeTriFromPointFeatureSets
	if(featureCoordinates.shape[0] >= ATORmaxNumberOfNearestFeaturesToSamplePolyKeypoints):
		sampleKeypointCoordinates = featureCoordinates	#first keypoint in candidate poly
		nearestKeypointCoordinates = ATORpt_operations.knn_search(sampleKeypointCoordinates, ATORmaxNumberOfNearestFeaturesToSamplePolyKeypoints)
		keypointSetList = []
		for k2 in range(ATORmaxNumberOfNearestFeaturesToSamplePolyKeypoints):
			for k3 in range(ATORmaxNumberOfNearestFeaturesToSamplePolyKeypoints):
				if(k2 != k3):
					keypointsSet = pt.stack((sampleKeypointCoordinates, nearestKeypointCoordinates[:, k2], nearestKeypointCoordinates[:, k3]), dim=1)
					keypointsSetMinX = pt.min(keypointsSet[..., xAxisFeatureMap], dim=1)
					keypointsSetMinY = pt.min(keypointsSet[..., yAxisFeatureMap], dim=1)
					keypointsSetMaxX = pt.max(keypointsSet[..., xAxisFeatureMap], dim=1)
					keypointsSetMaxY = pt.max(keypointsSet[..., yAxisFeatureMap], dim=1)
					keypointsSetMidX = mid(keypointsSet[..., xAxisFeatureMap], keypointsSetMinX, keypointsSetMaxX, dim=1)
					keypointsSetMidY = mid(keypointsSet[..., yAxisFeatureMap], keypointsSetMinY, keypointsSetMaxY, dim=1)
					
					if(keypointDetectionCriteria):
						#criteria 1: sufficient x and y diff in keypoints;
						keypointsSetXdiff = pt.subtract(keypointsSetMaxX.values, keypointsSetMinX.values)
						keypointsSetYdiff = pt.subtract(keypointsSetMaxY.values, keypointsSetMinY.values)
						maskXYdiff = pt.logical_and((keypointsSetXdiff > keypointDetectionMinXYdiff), (keypointsSetYdiff > keypointDetectionMinXYdiff))
						#criteria 3: ensure keypoints are not colinear
						maskNotColinear = keypointDetectionNotColinear(keypointsSet)
						if(fullRotationalInvariance):
							mask = pt.logical_and(maskXYdiff, maskNotColinear)
						else:
							#criteria 2: object triangle apex has sufficient y diff;
							maskApexYdiff = (keypointsSetMaxY.values >= (keypointsSetMidY['values']+keypointDetectionMinApexYDiff))
							mask = pt.logical_and(pt.logical_and(maskXYdiff, maskApexYdiff), maskNotColinear)
						keypointsSet = keypointsSet[mask]
						
					keypointSetList.append(keypointsSet)
		keypointCoordinates = pt.cat(keypointSetList, dim=0)		
	else:
		keypointCoordinates = pt.tensor((0, 3, 2))
	return keypointCoordinates

def keypointDetectionNotSame(keypoints):
	distances = pt.cdist(keypoints, keypoints)
	distances.fill_diagonal_(-1)
	distancesSimilar = (distances < keypointDetectionMaxSimilarity) & (distances != -1)
	distancesSimilar = distancesSimilar.float() * pt.triu(pt.ones_like(distancesSimilar, dtype=pt.float32), diagonal=1)
	distancesAnySimilar = pt.sum(distancesSimilar, dim=1)
	distancesAnySimilar = distancesAnySimilar > 0
	distancesAllDifferent = pt.logical_not(distancesAnySimilar)
	return distancesAllDifferent

def keypointDetectionNotColinear(keypoints):
	slope1 = (keypoints[:, 1, 1] - keypoints[:, 0, 1]) / (keypoints[:, 1, 0] - keypoints[:, 0, 0])
	slope2 = (keypoints[:, 2, 1] - keypoints[:, 1, 1]) / (keypoints[:, 2, 0] - keypoints[:, 1, 0])
	slope_diff = pt.abs(slope1 - slope2)
	non_colinear_mask = slope_diff >= keypointDetectionMaxColinearity
	return non_colinear_mask

def selectOtherKeypoints(keypointsSet, keypointsNotToSelect, axis):
	mask = pt.ones_like(keypointsSet[..., axis], dtype=pt.bool)
	mask[pt.arange(keypointsSet.shape[0]), keypointsNotToSelect.indices] = False
	resampled_tensor = pt.masked_select(keypointsSet[..., axis], mask).reshape(keypointsSet.shape[0], 2)
	return resampled_tensor
	
def mid(array, arrayMin, arrayMax, dim=1):
	min_indices = arrayMin.indices
	max_indices = arrayMax.indices	
	
	#alternate algorithm;
	#if minMaxIndicesSum = 0+1: then mid=2
	#if minMaxIndicesSum = 0+2: then mid=1
	#if minMaxIndicesSum = 1+2: then mid=0
	minMaxIndicesSum = pt.sum(pt.stack((min_indices, max_indices), dim=1), dim=1)
	mid_indices = pt.remainder(minMaxIndicesSum, 3).bool().int() * (pt.remainder(minMaxIndicesSum, 2) + 1)
	mid_values = array[pt.arange(array.shape[0]), mid_indices]
	result = {'values':mid_values, 'indices':mid_indices}
	'''
	mask = pt.ones(array.size(), dtype=pt.bool)
	pt.scatter(mask, dim, min_indices.unsqueeze(dim), 0)
	pt.scatter(mask, dim, max_indices.unsqueeze(dim), 0)
	mid_values = pt.masked_select(array, mask)
	'''
	#print("min_values = ", arrayMin.values)
	#print("max_values = ", arrayMax.values)
	#print("mid_values = ", mid_values)
	
	return result


	
