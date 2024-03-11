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
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
import cv2
import math

from ATORpt_globalDefs import *
import ATORpt_operations

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
	currentType = x.type()
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
	print("keypointCoordinates.shape = ", keypointCoordinates.shape)
	return keypointCoordinates
			
def performKeypointDetectionBasic(featureCoordinates):
	#based on ATORpt_RFapply:generateRFtypeTriFromPointFeatureSets
	if(featureCoordinates.shape[0] >= ATORmaxNumberOfNearestFeaturesToSamplePolyKeypoints):
		sampleKeypointCoordinates = featureCoordinates	#first keypoint in candidate poly
		nearestKeypointCoordinates = ATORpt_operations.knn_search(sampleKeypointCoordinates, ATORmaxNumberOfNearestFeaturesToSamplePolyKeypoints)
		keypointSetList = []
		for k2 in range(ATORmaxNumberOfNearestFeaturesToSamplePolyKeypoints):
			for k3 in range(ATORmaxNumberOfNearestFeaturesToSamplePolyKeypoints):
				if(k2 != k3):
					keypointsSet = pt.stack((sampleKeypointCoordinates, nearestKeypointCoordinates[:, k2], nearestKeypointCoordinates[:, k3]), dim=1)
					keypointsSetMinX = pt.min(keypointsSet[..., xAxisFeatureMap], dim=1).values
					keypointsSetMinY = pt.min(keypointsSet[..., yAxisFeatureMap], dim=1).values
					keypointsSetMaxX = pt.max(keypointsSet[..., xAxisFeatureMap], dim=1).values
					keypointsSetMaxY = pt.max(keypointsSet[..., yAxisFeatureMap], dim=1).values
					keypointsSetXdiff = pt.subtract(keypointsSetMaxX, keypointsSetMinX)
					keypointsSetYdiff = pt.subtract(keypointsSetMaxY, keypointsSetMinY)
					mask = pt.logical_and((keypointsSetXdiff > keypointDetectionMinXYdiff), (keypointsSetYdiff > keypointDetectionMinXYdiff))
					keypointsSet = keypointsSet[mask]
					keypointSetList.append(keypointsSet)
		keypointCoordinates = pt.cat(keypointSetList, dim=0)		
	else:
		keypointCoordinates = pt.tensor((0, 3, 2))
	print("keypointCoordinates.shape = ", keypointCoordinates.shape)
	return keypointCoordinates

def getSnapshotMeshCoordinates(keypointCoordinates, imagePath):
	#resample snapshot mesh coordinates wrt keypoints
	#FUTURE: this function needs to be upgraded to perform the image resampling in parallel
	
	image = cv2.imread(imagePath)
	image = TF.to_pil_image(image)
	
	snapshotPixelCoordinatesList = []
	snapshotMeshCoordinatesList = []	#mesh representation is offset by 0.5 pixels, size+1	#vertices
	snapshotMeshValuesList = []	#RGB values for each pixel/mesh coordinate in mesh	#colors
	snapshotMeshFacesList = []
	snapshotMeshPolyCoordinatesList = []
		
	for polyIndex in range(keypointCoordinates.shape[0]):
		polyKeypointCoordinates = keypointCoordinates[polyIndex]
		xMin = pt.min(polyKeypointCoordinates[:, 0])
		yMin = pt.min(polyKeypointCoordinates[:, 1])
		xMax = pt.max(polyKeypointCoordinates[:, 0])
		yMax = pt.max(polyKeypointCoordinates[:, 1])
		x = int((xMax - xMin) * ATORpatchPadding)
		y = int((yMax - yMin) * ATORpatchPadding)
		if(ATORpatchPadding > 1):
			xMinPadded = int(xMin-(x*ATORpatchPadding//2))
			yMinPadded = int(yMin-(y*ATORpatchPadding//2))
			croppedImage = TF.crop(image, yMinPadded, xMinPadded, y, x)
		else:
			xMinPadded = int(xMin)
			yMinPadded = int(yMin)
			croppedImage = image
		xSnapshot = normaliseSnapshotLength*ATORpatchPadding*ATORpatchUpscaling
		ySnapshot = normaliseSnapshotLength*ATORpatchPadding*ATORpatchUpscaling
		xScaling = x/xSnapshot
		yScaling = y/ySnapshot
		resampledImage = TF.resize(croppedImage, (ySnapshot, xSnapshot))
		resampledImage = ATORpt_operations.pil_to_tensor(resampledImage)
		
		xMesh = xSnapshot+1
		yMesh = ySnapshot+1
		snapshotPixelCoordinates = generatePixelCoordinates(xSnapshot, ySnapshot, xScaling, yScaling, xMinPadded, yMinPadded)
		snapshotMeshCoordinates = generatePixelCoordinates(xMesh, yMesh, xScaling, yScaling, xMinPadded, yMinPadded)
		snapshotMeshValues, snapshotMeshFaces = generatePixelValues(resampledImage, snapshotPixelCoordinates, xSnapshot, ySnapshot, xMesh, yMesh) 
		snapshotMeshPolyCoordinates = generateMeshPolyCoordinates(snapshotMeshCoordinates)
		
		snapshotPixelCoordinatesList.append(snapshotPixelCoordinates)
		snapshotMeshCoordinatesList.append(snapshotMeshCoordinates)
		snapshotMeshValuesList.append(snapshotMeshValues)
		snapshotMeshFacesList.append(snapshotMeshFaces)
		snapshotMeshPolyCoordinatesList.append(snapshotMeshPolyCoordinates)
	
	snapshotPixelCoordinates = pt.stack(snapshotPixelCoordinatesList, dim=0).to(device)
	snapshotMeshCoordinates = pt.stack(snapshotMeshCoordinatesList, dim=0).to(device)
	snapshotMeshValues = pt.stack(snapshotMeshValuesList, dim=0).to(device)
	snapshotMeshFaces = pt.stack(snapshotMeshFacesList, dim=0).to(device)
	snapshotMeshPolyCoordinates = pt.stack(snapshotMeshPolyCoordinatesList, dim=0).to(device)

	#print("snapshotPixelCoordinates.shape = ", snapshotPixelCoordinates.shape)
	#print("snapshotMeshCoordinates.shape = ", snapshotMeshCoordinates.shape)
	#print("snapshotMeshValues.shape = ", snapshotMeshValues.shape)
	#print("snapshotMeshFaces.shape = ", snapshotMeshFaces.shape)
	#print("snapshotMeshFaces.shape = ", snapshotMeshPolyCoordinates.shape)
	
	return snapshotPixelCoordinates, snapshotMeshCoordinates, snapshotMeshValues, snapshotMeshFaces, snapshotMeshPolyCoordinates
	
def generatePixelCoordinates(x, y, xScaling, yScaling, xMin, yMin):
	resampledImageShape = (y, x)
	polyPixelCoordinatesX = pt.arange(resampledImageShape[1]).expand(resampledImageShape[0], -1)	
	polyPixelCoordinatesY = pt.arange(resampledImageShape[0]).unsqueeze(1).expand(-1, resampledImageShape[1])
	polyPixelCoordinates = pt.stack((polyPixelCoordinatesX, polyPixelCoordinatesY), dim=-1)
	#print("polyPixelCoordinates.shape = ", polyPixelCoordinates.shape)
	polyPixelCoordinates[:, 0] = polyPixelCoordinates[:, 0] * yScaling
	polyPixelCoordinates[:, 1] = polyPixelCoordinates[:, 1] * xScaling
	polyPixelCoordinates[:, 0] = polyPixelCoordinates[:, 0] + yMin
	polyPixelCoordinates[:, 1] = polyPixelCoordinates[:, 1] + xMin
	polyPixelCoordinates = pt.reshape(polyPixelCoordinates, (polyPixelCoordinates.shape[0]*polyPixelCoordinates.shape[1], polyPixelCoordinates.shape[2]))
	return polyPixelCoordinates
	
def generatePixelValues(resampledImage, snapshotPixelCoordinates, xSnapshot, ySnapshot, xMesh, yMesh):
	numberOfVertices = snapshotPixelCoordinates.shape[0]
	#print("resampledImage.shape = ", resampledImage.shape)
	if(snapshotRenderExpectColorsDefinedForVerticesNotFaces):
		p2d = (0, 1, 0, 1) # pad last and second last dim by (0, 1)
		resampledImage = F.pad(resampledImage, p2d, "constant", snapshotRenderExpectColorsDefinedForVerticesNotFacesPadVal)
	#print("resampledImage.shape = ", resampledImage.shape)
	resampledImage = resampledImage.permute(2, 1, 0)	#(1, 2, 0)
	snapshotMeshValues = pt.reshape(resampledImage, (resampledImage.shape[0]*resampledImage.shape[1], resampledImage.shape[2]))	#reshape from c, y, x (cv2/TF/Pil) to x, y, c (pytorch3d)
	polyMeshCoordinateIndices = pt.arange(xMesh * yMesh).reshape(xMesh, yMesh)
	if(snapshotRenderTris):
		snapshotMeshFaces1a = polyMeshCoordinateIndices[0:xSnapshot, 0:ySnapshot]
		snapshotMeshFaces2a = polyMeshCoordinateIndices[1:xSnapshot+1, 0:ySnapshot]
		snapshotMeshFaces3a = polyMeshCoordinateIndices[0:xSnapshot, 1:ySnapshot+1]
		snapshotMeshFaces1b = polyMeshCoordinateIndices[1:xSnapshot+1, 0:ySnapshot]
		snapshotMeshFaces2b = polyMeshCoordinateIndices[0:xSnapshot, 1:ySnapshot+1]
		snapshotMeshFaces3b = polyMeshCoordinateIndices[1:xSnapshot+1, 1:ySnapshot+1]
		snapshotMeshFaces1 = pt.stack((snapshotMeshFaces1a, snapshotMeshFaces2a, snapshotMeshFaces3a), dim=2)
		snapshotMeshFaces1 = pt.reshape(snapshotMeshFaces1, (xSnapshot*ySnapshot, snapshotMeshFaces1.shape[2]))
		snapshotMeshFaces2 = pt.stack((snapshotMeshFaces1b, snapshotMeshFaces2b, snapshotMeshFaces3b), dim=2)
		snapshotMeshFaces2 = pt.reshape(snapshotMeshFaces2, (xSnapshot*ySnapshot, snapshotMeshFaces2.shape[2]))
		snapshotMeshFaces = pt.cat((snapshotMeshFaces1, snapshotMeshFaces2), dim=0)
		if(not snapshotRenderExpectColorsDefinedForVerticesNotFaces):
			snapshotMeshValues = pt.cat((snapshotMeshValues, snapshotMeshValues), dim=0)
	else:
		snapshotMeshFaces1 = polyMeshCoordinateIndices[0:xSnapshot, 0:ySnapshot]
		snapshotMeshFaces2 = polyMeshCoordinateIndices[1:xSnapshot+1, 0:ySnapshot]
		snapshotMeshFaces3 = polyMeshCoordinateIndices[0:xSnapshot, 1:ySnapshot+1]
		snapshotMeshFaces4 = polyMeshCoordinateIndices[1:xSnapshot+1, 1:ySnapshot+1]
		snapshotMeshFaces = pt.stack((snapshotMeshFaces1, snapshotMeshFaces2, snapshotMeshFaces3, snapshotMeshFaces4), dim=2)
		snapshotMeshFaces = pt.reshape(snapshotMeshFaces, (xSnapshot*ySnapshot, snapshotMeshFaces.shape[2]))
		#snapshotMeshPolyCoordinates = snapshotPixelCoordinates[snapshotMeshFaces]	#this causes CUDA error
			
	return snapshotMeshValues, snapshotMeshFaces

def generateMeshPolyCoordinates(snapshotMeshCoordinates):
	snapshotMeshCoordinatesLength = int(math.sqrt(snapshotMeshCoordinates.shape[0]))
	snapshotPixelCoordinatesLength = snapshotMeshCoordinatesLength-1
	snapshotMeshCoordinates = pt.reshape(snapshotMeshCoordinates, (snapshotMeshCoordinatesLength, snapshotMeshCoordinatesLength, snapshotMeshCoordinates.shape[1]))
	snapshotMeshPolyCoordinates1 = snapshotMeshCoordinates[0:snapshotPixelCoordinatesLength, 0:snapshotPixelCoordinatesLength, :]
	snapshotMeshPolyCoordinates2 = snapshotMeshCoordinates[1:snapshotPixelCoordinatesLength+1, 0:snapshotPixelCoordinatesLength, :]
	snapshotMeshPolyCoordinates3 = snapshotMeshCoordinates[0:snapshotPixelCoordinatesLength, 1:snapshotPixelCoordinatesLength+1, :]
	snapshotMeshPolyCoordinates4 = snapshotMeshCoordinates[1:snapshotPixelCoordinatesLength+1, 1:snapshotPixelCoordinatesLength+1, :]
	snapshotMeshPolyCoordinates = pt.stack((snapshotMeshPolyCoordinates1, snapshotMeshPolyCoordinates2, snapshotMeshPolyCoordinates3, snapshotMeshPolyCoordinates4), dim=2)
	snapshotMeshPolyCoordinates = pt.reshape(snapshotMeshPolyCoordinates, (snapshotPixelCoordinatesLength*snapshotPixelCoordinatesLength, snapshotMeshPolyCoordinates.shape[2], snapshotMeshPolyCoordinates.shape[3]))
	return snapshotMeshPolyCoordinates

def getImageMeshCoordinates(imagePath):
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = TF.to_pil_image(image)
	image_size = image.size
	width, height = image_size
	
	'''
	#xMin = width//2
	#yMin = height//2
	#xSnapshot = 8
	#ySnapshot = 8
	'''
	'''
	xMin = 0
	yMin = 0
	xSnapshot = height
	ySnapshot = height
	'''
	xMin = 200
	yMin = 200
	xSnapshot = 300
	ySnapshot = 300
	print("height = ", height)

	image = TF.crop(image, yMin, xMin, ySnapshot, xSnapshot)	#make sure image is square
	image = ATORpt_operations.pil_to_tensor(image)
	
	print("image.shape = ", image.shape)
	
	xMin = 0	#set start coordinates 0 zero for immediate rendering (without geohashing)
	yMin = 0 	#set start coordinates 0 zero for immediate rendering (without geohashing)
	xMesh = xSnapshot+1
	yMesh = ySnapshot+1	
	xScaling = 1.0
	yScaling = 1.0
	snapshotPixelCoordinates = generatePixelCoordinates(xSnapshot, ySnapshot, xScaling, yScaling, xMin, yMin)
	snapshotMeshCoordinates = generatePixelCoordinates(xMesh, yMesh, xScaling, yScaling, xMin, yMin)
	snapshotMeshValues, snapshotMeshFaces = generatePixelValues(image, snapshotPixelCoordinates, xSnapshot, ySnapshot, xMesh, yMesh) 
	#print("snapshotMeshCoordinates.shape = ", snapshotMeshCoordinates.shape)
	snapshotMeshPolyCoordinates = generateMeshPolyCoordinates(snapshotMeshCoordinates)

	snapshotPixelCoordinates = snapshotPixelCoordinates.unsqueeze(0)
	snapshotMeshCoordinates = snapshotMeshCoordinates.unsqueeze(0)
	snapshotMeshValues = snapshotMeshValues.unsqueeze(0)
	snapshotMeshFaces = snapshotMeshFaces.unsqueeze(0)
	snapshotMeshPolyCoordinates = snapshotMeshPolyCoordinates.unsqueeze(0)
		
	return snapshotPixelCoordinates, snapshotMeshCoordinates, snapshotMeshValues, snapshotMeshFaces, snapshotMeshPolyCoordinates
		
