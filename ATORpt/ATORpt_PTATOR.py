"""ATORpt_PTATOR.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt PT ATOR -  parallel processing of ATOR (third party feature detection and geometric hashing)

"""

import torch as pt
import cv2

from ATORpt_globalDefs import *
import ATORpt_PTfeatureDetector
import ATORpt_PTpolyKeypointGenerator
import ATORpt_PTgeometricHashing
import ATORpt_operations
import ATORpt_PTrenderer

def generateATORpatches(imagePaths, train):

	if(snapshotRenderDebug):
		snapshotPixelCoordinates, snapshotMeshCoordinates, snapshotMeshValues, snapshotMeshFaces, snapshotMeshPolyCoordinates = ATORpt_PTpolyKeypointGenerator.getImageMeshCoordinates(imagePaths[0])
		renderImageSize = 500
		transformedPatches = ATORpt_PTrenderer.resamplePixelCoordinates(snapshotMeshCoordinates, snapshotMeshValues, snapshotMeshFaces, renderImageSize)	#transformedSnapshotPixelCoordinates
	
	keypointCoordinatesList = []
	snapshotPixelCoordinatesList = []
	snapshotMeshCoordinatesList = []
	snapshotMeshValuesList = []
	snapshotMeshFacesList = []
	snapshotMeshPolyCoordinatesList = []
	
	for imageIndex, imagePath in enumerate(imagePaths):
		print("imageIndex = ", imageIndex)
		imageKeypointCoordinatesZList = []
		for zoomIndex in range(numberOfZoomLevels):
			print("zoomIndex = ", zoomIndex)
			image = getImageCV(imagePath, zoomIndex)	#shape = [Height, Width, Channels] where Channels is [Blue, Green, Red] (opencv standard)
			imageFeatureCoordinatesZ = ATORpt_PTfeatureDetector.featureDetection(image, zoomIndex)
			imageKeypointCoordinatesZ = ATORpt_PTpolyKeypointGenerator.performKeypointDetection(imageFeatureCoordinatesZ)
			imageKeypointCoordinatesZList.append(imageKeypointCoordinatesZ)
		imageKeypointCoordinates = pt.cat(imageKeypointCoordinatesZList, dim=0)
		imageKeypointCoordinates = ATORpt_PTpolyKeypointGenerator.cropCoordinatesArray(imageKeypointCoordinates)	#crop imageKeypointCoordinates to ATORmaxNumberOfPolys
		snapshotPixelCoordinates, snapshotMeshCoordinates, snapshotMeshValues, snapshotMeshFaces, snapshotMeshPolyCoordinates = ATORpt_PTpolyKeypointGenerator.getSnapshotMeshCoordinates(imageKeypointCoordinates, imagePath)
		imageKeypointCoordinates, snapshotPixelCoordinates, snapshotMeshCoordinates, snapshotMeshValues, snapshotMeshFaces, snapshotMeshPolyCoordinates = ATORpt_PTpolyKeypointGenerator.padCoordinatesArrays(imageKeypointCoordinates, snapshotPixelCoordinates, snapshotMeshCoordinates, snapshotMeshValues, snapshotMeshFaces, snapshotMeshPolyCoordinates)	#pad coordinates to ATORmaxNumberOfPolys

		keypointCoordinatesList.append(imageKeypointCoordinates)
		snapshotPixelCoordinatesList.append(snapshotPixelCoordinates)
		snapshotMeshCoordinatesList.append(snapshotMeshCoordinates)
		snapshotMeshValuesList.append(snapshotMeshValues)
		snapshotMeshFacesList.append(snapshotMeshFaces)	
		snapshotMeshPolyCoordinatesList.append(snapshotMeshPolyCoordinates)	
		
	keypointCoordinates = pt.cat(keypointCoordinatesList, dim=0)
	snapshotPixelCoordinates = pt.cat(snapshotPixelCoordinatesList, dim=0)
	snapshotMeshCoordinates = pt.cat(snapshotMeshCoordinatesList, dim=0)
	snapshotMeshValues = pt.cat(snapshotMeshValuesList, dim=0)
	snapshotMeshFaces = pt.cat(snapshotMeshFacesList, dim=0)
	snapshotMeshPolyCoordinates = pt.cat(snapshotMeshPolyCoordinatesList, dim=0)
	
	print("snapshotPixelCoordinates.shape = ", snapshotPixelCoordinates.shape)
	print("snapshotMeshCoordinates.shape = ", snapshotMeshCoordinates.shape)
	print("snapshotMeshValues.shape = ", snapshotMeshValues.shape)
	print("snapshotMeshFaces.shape = ", snapshotMeshFaces.shape)
	print("snapshotMeshPolyCoordinates.shape = ", snapshotMeshPolyCoordinates.shape)

	if(debugGeometricHashingParallel):
		pixelCoordinates = snapshotPixelCoordinates
	else:
		if(snapshotRenderer == "torchgeometry"):
			pixelCoordinates = snapshotMeshPolyCoordinates
		else:
			pixelCoordinates = snapshotMeshCoordinates
	transformedSnapshotPixelCoordinates = ATORpt_PTgeometricHashing.performGeometricHashingParallel(keypointCoordinates, pixelCoordinates, pixelValues=snapshotMeshValues)	
	renderImageSize = normaliseSnapshotLength*ATORpatchPadding
	transformedPatches = ATORpt_PTrenderer.resamplePixelCoordinates(snapshotMeshCoordinates, snapshotMeshValues, snapshotMeshFaces, renderImageSize)	#transformedSnapshotPixelCoordinates
	return transformedPatches

def getImageCV(imagePath, zoomIndex):
	zoom = ATORpt_PTfeatureDetector.getZoomValue(zoomIndex)
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	imageWidth = image.shape[1] // zoom
	imageHeight = image.shape[0] // zoom
	image = cv2.resize(image, (imageWidth, imageHeight))
	return image


