"""ATORpt_PTgeometricHashing3DOD.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt PT geometric Hashing 3DOD - parallel processing of ATOR geometric hashing for 3D object data

"""

import torch as pt
import math
#from pytorch3d.transforms import rotation_conversions
#from pytorch3d.renderer.cameras import look_at_view_transform
from pytorch3d.renderer.cameras import look_at_rotation


from ATORpt_globalDefs import *
import ATORpt_operations

def performGeometricHashingParallel(keypointCoordinates, meshCoordinates, meshValues=None, meshFaces=None):
	use3DOD = True
	#keypointCoordinates shape: [numberPolys, snapshotNumberOfKeypoints, numberOfGeometricDimensions3DOD], i.e. [yCoordinate, xCoordinate, zCoordinate] for each pixel in meshCoordinates
	#meshCoordinates shape: [numberPolys, numberCoordinatesInSnapshot, numberOfGeometricDimensions3DOD], i.e. [yCoordinate, xCoordinate, zCoordinate] for each keypoint in poly
	
	print("start performGeometricHashingParallel")

	#based on https://patentscope.wipo.int/search/en/detail.jsf?docId=WO2011088497 Fig 30->35
	#see ATORmethod3DOD:transformObjectData3DOD for unvectorised (ie serial processing) method
	
	transformedMeshCoordinates = meshCoordinates
	transformedKeypointCoordinates = keypointCoordinates	#retain original for calculations	#.copy()
	ATORpt_operations.printCoordinatesIndex(use3DOD, transformedKeypointCoordinates, transformedMeshCoordinates, meshValues, meshFaces, index=debugPolyIndex, step=0)	#will be out of range of render view port
	#ATORpt_operations.printCoordinates(use3DOD, transformedKeypointCoordinates, transformedMeshCoordinates, meshValues, meshFaces, step=0, centreSnapshots=False)
	
	#apply hardcoded geometric hashing function;

	if(use3DODgeoHashingOrig):
		#step 1. Perform All Rotations (X, Y, Z, such that object triangle side is parallel with x axis, and object triangle normal is parallel with z axis)
		normalBeforeRotation = calculateNormalOfTri(transformedKeypointCoordinates[:, 0], transformedKeypointCoordinates[:, 1], transformedKeypointCoordinates[:, 2])
		normalBeforeRotationNormalised = normaliseVector(normalBeforeRotation)
		batchSize = keypointCoordinates.shape[0] 
		eye = pt.zeros([batchSize, numberOfGeometricDimensions3DOD], device=device)
		#eye = calculateMidPointBetweenTwoPoints(transformedKeypointCoordinates[:,0],transformedKeypointCoordinates[:,1])	#check this (use keypoints 1 and 2?)
		at = normalBeforeRotationNormalised
		up = subtractVectors(transformedKeypointCoordinates[:, 0], transformedKeypointCoordinates[:, 1])	#check this (use keypoints 1 and 2?)
		R = generateLookAtRotationMatrix(at, eye, up)
		R = pt.transpose(R, 1, 2)
		transformedMeshCoordinates = pt.matmul(R, transformedMeshCoordinates.transpose(1, 2)).transpose(1, 2)
		transformedKeypointCoordinates = pt.matmul(R, transformedKeypointCoordinates.transpose(1, 2)).transpose(1, 2)	
		
		#step 2. Translate the object data on all axes such that the mid point of the object triangle side passes through the Z axis coordinate 0.
		translationVector = calculateMidPointBetweenTwoPoints(transformedKeypointCoordinates[:,0],transformedKeypointCoordinates[:,1])	#check this (use keypoints 1 and 2?)
		translationVector = pt.unsqueeze(translationVector, 1)
		transformedMeshCoordinates = transformedMeshCoordinates - translationVector
		transformedKeypointCoordinates = transformedKeypointCoordinates - translationVector
		if(snapshotRenderCameraZworkaround):
			transformedMeshCoordinates[:, :, zAxisGeometricHashing] = snapshotRenderZdimVal
			transformedKeypointCoordinates[:, :, zAxisGeometricHashing] = snapshotRenderZdimVal
		ATORpt_operations.printCoordinatesIndex(use3DOD, transformedKeypointCoordinates, transformedMeshCoordinates, meshValues, meshFaces, index=debugPolyIndex, step=2)
	
		if(use3DODgeoHashingScale):
			#added: rescale image to align with viewport
			#step 3 (scale x - wrt object triangle [0, 1]):   
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
	else:
		#based on 2DOD method;

		#keypointCoordinates;
		# kp2
		#  \\
		#   \_\
		#  kp1 kp0  
	
		#step 1 (shift x/y - wrt centre of keypointCoordinates [0, 1]):
		#translate object data on X and Y axis such that the object triangle base is positioned at centre of keypointCoordinates [0, 1]):
		#Fig 31
		keypointsTriBaseCentre = pt.add(keypointCoordinates[:, 0], keypointCoordinates[:, 1])/2.0
		transformedMeshCoordinates = pt.subtract(transformedMeshCoordinates, keypointsTriBaseCentre.unsqueeze(1))
		transformedKeypointCoordinates = pt.subtract(transformedKeypointCoordinates, keypointsTriBaseCentre.unsqueeze(1))
		ATORpt_operations.printCoordinatesIndex(use3DOD, transformedKeypointCoordinates, transformedMeshCoordinates, meshValues, meshFaces, index=debugPolyIndex, step=1)

		#step 2. Perform All Rotations (X, Y, Z, such that object triangle side is parallel with x axis, and object triangle normal is parallel with z axis)
		normalBeforeRotation = calculateNormalOfTri(transformedKeypointCoordinates[:, 0], transformedKeypointCoordinates[:, 1], transformedKeypointCoordinates[:, 2])
		normalBeforeRotationNormalised = normaliseVector(normalBeforeRotation)
		batchSize = keypointCoordinates.shape[0] 
		eye = pt.zeros([batchSize, numberOfGeometricDimensions3DOD], device=device)
		#eye = calculateMidPointBetweenTwoPoints(transformedKeypointCoordinates[:,0],transformedKeypointCoordinates[:,1])	#check this (use keypoints 1 and 2?)
		at = normalBeforeRotationNormalised
		up = subtractVectors(transformedKeypointCoordinates[:, 0], transformedKeypointCoordinates[:, 1])	#check this (use keypoints 1 and 2?)
		R = generateLookAtRotationMatrix(at, eye, up)
		R = pt.transpose(R, 1, 2)
		transformedMeshCoordinates = pt.matmul(R, transformedMeshCoordinates.transpose(1, 2)).transpose(1, 2)
		transformedKeypointCoordinates = pt.matmul(R, transformedKeypointCoordinates.transpose(1, 2)).transpose(1, 2)
	
		#R = look_at_rotation(eye, at, up, device=device)	#https://pytorch3d.readthedocs.io/en/latest/modules/renderer/cameras.html#pytorch3d.renderer.cameras.look_at_rotation
		#transformedMeshCoordinates = pt.matmul(R, transformedMeshCoordinates.transpose(1, 2)).transpose(1, 2)
		#transformedKeypointCoordinates = pt.matmul(R, transformedKeypointCoordinates.transpose(1, 2)).transpose(1, 2)
		#R, T = look_at(eye=eye, at=at, up=up)	#look_at_view_transform

		#R = look_at_rotation(eye, at, up, device=device)	#https://pytorch3d.readthedocs.io/en/latest/modules/renderer/cameras.html#pytorch3d.renderer.cameras.look_at_rotation
		#transformedMeshCoordinates = rotate_coordinates_batch(transformedMeshCoordinates, R)
		#transformedKeypointCoordinates = rotate_coordinates_batch(transformedKeypointCoordinates, R)

		if(snapshotRenderCameraZworkaround):
			transformedMeshCoordinates[:, :, zAxisGeometricHashing] = snapshotRenderZdimVal
			transformedKeypointCoordinates[:, :, zAxisGeometricHashing] = snapshotRenderZdimVal
			
		ATORpt_operations.printCoordinatesIndex(use3DOD, transformedKeypointCoordinates, transformedMeshCoordinates, meshValues, meshFaces, index=debugPolyIndex, step=2)

		if(use3DODgeoHashingScale):
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

	
	#ATORpt_operations.printCoordinates(use3DOD, transformedKeypointCoordinates, transformedMeshCoordinates, meshValues, meshFaces, step=5, centreSnapshots=False)

	return transformedMeshCoordinates

def generateLookAtRotationMatrix(at, eye, up):
	zaxis = subtractVectors(at, eye)
	zaxis = normaliseVector(zaxis)
	xaxis = crossProduct(up, zaxis)
	xaxis = normaliseVector(xaxis)
	yaxis = crossProduct(zaxis, xaxis)
	rotationMatrix = makeMatrix(xaxis, yaxis, zaxis)
	return rotationMatrix

def crossProduct(vect1, vect2):
	vect = pt.cross(vect1, vect2)
	return vect

def makeMatrix(xaxis, yaxis, zaxis):
	matx = pt.stack((xaxis, yaxis, zaxis), dim=1)
	return matx

def rotate_coordinates_batch(coordinates, R):
	coordinates_expanded = coordinates.unsqueeze(-2)  # (batchSize, numCoordinates, 1, 3)
	rotated_coordinates = pt.matmul(coordinates_expanded, R.unsqueeze(1))  # (batchSize, 1, numCoordinates, 3)
	rotated_coordinates = rotated_coordinates.squeeze(-2)  # (batchSize, numCoordinates, 3)
	return rotated_coordinates
	
'''
def look_at(eye, at, up):
	forward = at - eye
	forward = forward / pt.norm(forward, dim=1, keepdim=True)
	right = pt.cross(forward, up)
	right = right / pt.norm(right, dim=1, keepdim=True)
	up = pt.cross(right, forward)
	R = pt.stack([right, up, -forward], dim=2)
	T = -pt.matmul(R, eye.unsqueeze(2)).squeeze()
	return R, T
def rotate_coordinates_batch(coordinates, R):
	coordinates_expanded = coordinates.unsqueeze(1)  # (batchSize, 1, numCoordinates, 3)
	rotated_coordinates = pt.matmul(coordinates_expanded, R.unsqueeze(1))  # (batchSize, 1, numCoordinates, 3)
	rotated_coordinates = rotated_coordinates.squeeze(1)  # (batchSize, numCoordinates, 3)
	return rotated_coordinates
'''
'''
def rotate_coordinates_batch(coordinates, R):
	coordinates = coordinates.transpose(1, 2)  # Transpose to get [batchSize, 3, numberCoordinates]
	rotated_coordinates = torch.einsum('bij,bjk->bik', rotation_matrices, coordinates)
	rotated_coordinates = rotated_coordinates.transpose(1, 2)  # Back to [batchSize, numberCoordinates, 3]
	return rotated_coordinates
'''
def calculateNormalOfTri(pt1, pt2, pt3):
	vec1 = subtractVectors(pt2, pt1)
	vec2 = subtractVectors(pt3, pt1)
	normal = calculateNormal(vec1, vec2)
	return normal

def subtractVectors(vect1, vect2):
	vect = vect1 - vect2
	return vect

def normaliseVector(vect):
	magnitude = findMagnitudeOfVector(vect)
	vect = vect/magnitude.unsqueeze(-1)
	return vect

def findMagnitudeOfVector(vect):
	magnitude = pt.sqrt(pt.pow(vect[:, 0], 2) + pt.pow(vect[:, 1],2) + pt.pow(vect[:, 2],2))
	return magnitude

def calculateNormal(pt1, pt2):
	normal = pt.cross(pt1, pt2)
	return normal

def calculateMidPointBetweenTwoPoints(pt1, pt2):
	midDiff = calculateMidDiffBetweenTwoPoints(pt1, pt2)
	midPoint = pt1 + midDiff
	return midPoint

def calculateMidDiffBetweenTwoPoints(pt1, pt2):
	midDiff = (pt2 - pt1)/2.0
	return midDiff

def calculateDistance(keypoint1, keypoint2):
	#dist = sqrt((x1-x2)^2 + (y1-y2)^2) 
	xDiff = keypoint1[:, xAxisGeometricHashing] - keypoint2[:, xAxisGeometricHashing] 
	yDiff = keypoint1[:, yAxisGeometricHashing] - keypoint2[:, yAxisGeometricHashing] 
	zDiff = keypoint1[:, zAxisGeometricHashing] - keypoint2[:, zAxisGeometricHashing] 
	distance = pt.sqrt(pt.add(pt.square(xDiff), pt.add(pt.square(yDiff), pt.square(zDiff)))) 
	return distance
	
