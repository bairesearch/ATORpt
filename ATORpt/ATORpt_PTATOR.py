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
	#image polys (ie polys across all images) are merged into single dim for geometric hashing and rendering
	
	if(debugSnapshotRender):
		imagePath = imagePaths[0]
		snapshotPixelCoordinates, snapshotMeshCoordinates, snapshotMeshValues, snapshotMeshFaces, snapshotMeshPolyCoordinates = ATORpt_PTpolyKeypointGenerator.getImageMeshCoordinates(imagePath)
		renderViewportSizeImage = (500, 500)
		renderImageSizeImage = 500
		transformedPatches = ATORpt_PTrenderer.resamplePixelCoordinates(snapshotMeshCoordinates, snapshotMeshValues, snapshotMeshFaces, renderViewportSizeImage, renderImageSizeImage, centreSnapshots=True)	#transformedSnapshotPixelCoordinates
	
	keypointCoordinatesList = []
	snapshotPixelCoordinatesList = []
	snapshotMeshCoordinatesList = []
	snapshotMeshValuesList = []
	snapshotMeshFacesList = []
	snapshotMeshPolyCoordinatesList = []
	
	for imageIndex, imagePath in enumerate(imagePaths):
		#print("imageIndex = ", imageIndex)
		imageKeypointCoordinatesZList = []
		for zoomIndex in range(numberOfZoomLevels):
			#print("zoomIndex = ", zoomIndex)
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
	
	'''
	print("keypointCoordinates.shape = ", keypointCoordinates.shape)
	print("snapshotPixelCoordinates.shape = ", snapshotPixelCoordinates.shape)
	print("snapshotMeshCoordinates.shape = ", snapshotMeshCoordinates.shape)
	print("snapshotMeshValues.shape = ", snapshotMeshValues.shape)
	print("snapshotMeshFaces.shape = ", snapshotMeshFaces.shape)
	print("snapshotMeshPolyCoordinates.shape = ", snapshotMeshPolyCoordinates.shape)
	'''
	
	if(snapshotRenderer == "torchgeometry"):
		meshCoordinates = snapshotMeshPolyCoordinates
	elif(snapshotRenderer == "pytorch3D"):
		meshCoordinates = snapshotMeshCoordinates
	
	transformedMeshCoordinates = ATORpt_PTgeometricHashing.performGeometricHashingParallel(keypointCoordinates, meshCoordinates, meshValues=snapshotMeshValues, meshFaces=snapshotMeshFaces)
	print("geoHashing complete")
	transformedPatches = ATORpt_PTrenderer.resamplePixelCoordinates(transformedMeshCoordinates, snapshotMeshValues, snapshotMeshFaces, renderViewportSize, renderImageSize, centreSnapshots=True)	#transformedSnapshotPixelCoordinates	#after debug; centreSnapshots=False
	print("transformedPatches.shape = ", transformedPatches.shape)
	numPatchesPerImage = transformedPatches.shape[0]//batchSize
	transformedPatches = pt.reshape(transformedPatches, (batchSize, numPatchesPerImage, transformedPatches.shape[1], transformedPatches.shape[2], transformedPatches.shape[3]))	#shape: batchSize*ATORmaxNumberOfPolys, H, W, C
	transformedPatches = pt.permute(transformedPatches, (0, 1, 4, 2, 3))	#shape: batchSize, ATORmaxNumberOfPolys, C, H, W
	print("transformedPatches.shape = ", transformedPatches.shape)
	
	return transformedPatches

def getImageCV(imagePath, zoomIndex):
	zoom = ATORpt_PTfeatureDetector.getZoomValue(zoomIndex)
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	imageWidth = image.shape[1] // zoom
	imageHeight = image.shape[0] // zoom
	image = cv2.resize(image, (imageWidth, imageHeight))
	return image


