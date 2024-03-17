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

def reorderKeypoints(keypointCoordinates):

	#reorder keypointCoordinates;
	# kp2
	#  \\
	#   \_\
	#  kp1 kp0  
	
	keypointCoordinates = keypointSwap(keypointCoordinates, 2, 1, 1, yAxisGeometricHashing)
	keypointCoordinates = keypointSwap(keypointCoordinates, 2, 0, 0, yAxisGeometricHashing)
	keypointCoordinates = keypointSwap(keypointCoordinates, 0, 1, 1, xAxisGeometricHashing)
	
	return keypointCoordinates

def keypointSwap(keypoints, keypointAindex, keypointBindex, keypointCindex, axis):

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
		print("keypointSwap(keypointCoordinates, keypointAindex=", keypointAindex, ", keypointBindex=",  keypointBindex, ", keypointCindex=", keypointCindex, ", axis=", axis)
		print("keypoints[0] = ", keypoints[0])

	return keypoints
	
	
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

def cropCoordinatesArray(coordinates):
	dimToCrop = 0
	if(coordinates.shape[0] > ATORmaxNumberOfPolys):
		coordinates = coordinates[0:ATORmaxNumberOfPolys]
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
						#mask = maskXYdiff
						#criteria 2: object triangle apex has sufficient y diff;
						maskApexYdiff = (keypointsSetMaxY.values >= (keypointsSetMidY['values']+keypointDetectionMinApexYDiff))
						#criteria 3: ensure keypoints are not colinear
						maskNotColinear = keypointDetectionNotColinear(keypointsSet)
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


	
