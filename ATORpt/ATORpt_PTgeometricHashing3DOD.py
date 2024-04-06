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
from pytorch3d.renderer.cameras import look_at_view_transform

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
	#ATORpt_operations.printCoordinatesIndex(use3DOD, transformedKeypointCoordinates, transformedMeshCoordinates, meshValues, meshFaces, index=debugPolyIndex, step=0)	#will be out of range of render view port
	#ATORpt_operations.printCoordinates(use3DOD, transformedKeypointCoordinates, transformedMeshCoordinates, meshValues, meshFaces, step=0, centreSnapshots=False)
	
	#apply hardcoded geometric hashing function;

	#1. Perform All Rotations (X, Y, Z, such that object triangle side is parallel with x axis, and object triangle normal is parallel with z axis)
	normalBeforeRotation = calculateNormalOfTri(keypointCoordinates[:, 0], keypointCoordinates[:, 1], keypointCoordinates[:, 2])
	normalBeforeRotationNormalised = normaliseVector(normalBeforeRotation)
	batchSize = keypointCoordinates.shape[0] 
	eye = pt.zeros([batchSize, numberOfGeometricDimensions3DOD], device=device)
	at = normalBeforeRotationNormalised
	up = subtractVectors(keypointCoordinates[:, 0], keypointCoordinates[:, 1])	#check this (use keypoints 1 and 2?)
	
	#R = rotation_conversions.look_at_rotation(eye, at, up)
	#transformedMeshCoordinates = pt.matmul(R, transformedMeshCoordinates.transpose(1, 2)).transpose(1, 2)
	#transformedKeypointCoordinates = pt.matmul(R, transformedKeypointCoordinates.transpose(1, 2)).transpose(1, 2)
	R, T = look_at(eye=eye, at=at, up=up)	#look_at_view_transform
	transformedMeshCoordinates = rotate_coordinates_batch(transformedMeshCoordinates, R)
	transformedKeypointCoordinates = rotate_coordinates_batch(transformedKeypointCoordinates, R)
	
	#ATORpt_operations.printCoordinatesIndex(use3DOD, transformedKeypointCoordinates, transformedMeshCoordinates, meshValues, meshFaces, index=debugPolyIndex, step=1)

	#3a. Translate the object data on all axes such that the mid point of the object triangle side passes through the Z axis coordinate 0.
	translationVector = calculateMidPointBetweenTwoPoints(transformedKeypointCoordinates[:,0],transformedKeypointCoordinates[:,1])	#check this (use keypoints 1 and 2?)
	translationVector = pt.unsqueeze(translationVector, 1)
	transformedMeshCoordinates = transformedMeshCoordinates + translationVector
	transformedKeypointCoordinates = transformedKeypointCoordinates + translationVector
	#ATORpt_operations.printCoordinatesIndex(use3DOD, transformedKeypointCoordinates, transformedMeshCoordinates, meshValues, meshFaces, index=debugPolyIndex, step=2)

	#ATORpt_operations.printCoordinates(use3DOD, transformedKeypointCoordinates, transformedMeshCoordinates, meshValues, meshFaces, step=5, centreSnapshots=False)

	return transformedMeshCoordinates


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
