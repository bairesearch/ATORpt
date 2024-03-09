"""ATORpt_PTpolyKeypointGenerator.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt PT poly Keypoint Generator

"""

import torch as pt
import torch.nn as nn
import torch.nn.functional as F

from ATORpt_globalDefs import *
import ATORpt_operations

def padCoordinatesArrays(imageKeypointCoordinates, snapshotMeshCoordinates, snapshotPixelCoordinates, snapshotPixelValues, snapshotPixelFaces):
	#ensure num of polys is equal per image
	imageKeypointCoordinates = padCoordinatesArray(imageKeypointCoordinates)
	snapshotMeshCoordinates = padCoordinatesArray(snapshotMeshCoordinates)
	snapshotPixelCoordinates = padCoordinatesArray(snapshotPixelCoordinates)
	snapshotPixelValues = padCoordinatesArray(snapshotPixelValues)
	snapshotPixelFaces = padCoordinatesArray(snapshotPixelFaces)
	return imageKeypointCoordinates, snapshotMeshCoordinates, snapshotPixelCoordinates, snapshotPixelValues, snapshotPixelFaces
	
def padCoordinatesArray(coordinates):
	coordinatesPadded = padAlongDimension(coordinates, 0, ATORmaxNumberOfPolys)
	return coordinatesPadded

def padAlongDimension(x, dimToPad, targetSize):
	zerosShape = list(x.shape)
	zerosShape[dimToPad] = targetSize
	zeros = pt.zeros(zerosShape).to(device)
	xPadded = pt.cat((x, zeros), dim=dimToPad)
	return xPadded


def performKeypointDetection(imageFeatureCoordinates):
	#featureCoordinatesList size = batchSize list of [featureIndex, x/yIndex]
	keypointCoordinates = performKeypointDetectionBasic(imageFeatureCoordinates)
	print("keypointCoordinates.shape = ", keypointCoordinates.shape)
	return keypointCoordinates
			
def performKeypointDetectionBasic(featureCoordinates):
	#based on ATORpt_RFapply:generateRFtypeTriFromPointFeatureSets
	if(featureCoordinates.shape[0] >= snapshotNumberOfKeypoints):
		keypointCoordinates = ATORpt_operations.knn_search(featureCoordinates, snapshotNumberOfKeypoints)
		#TODO: requires check to ensure that poly keypoints are not colinear
	else:
		keypointCoordinates = pt.tensor((0, 3, 2))
	return keypointCoordinates

