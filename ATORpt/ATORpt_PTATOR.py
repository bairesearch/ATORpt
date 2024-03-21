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
import torchvision.transforms.functional as TF

from ATORpt_globalDefs import *
import ATORpt_PTfeatures
import ATORpt_PTkeypoints
import ATORpt_PTmesh
import ATORpt_PTgeometricHashing
import ATORpt_operations
import ATORpt_PTrenderer

def generateATORpatches(imagePaths, train):
	#image polys (ie polys across all images) are merged into single dim for geometric hashing and rendering
	
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
			imageFeatureCoordinatesZ = ATORpt_PTfeatures.featureDetection(image, zoomIndex)
			imageKeypointCoordinatesZ = ATORpt_PTkeypoints.performKeypointDetection(imageFeatureCoordinatesZ)
			imageKeypointCoordinatesZList.append(imageKeypointCoordinatesZ)
		imageKeypointCoordinates = pt.cat(imageKeypointCoordinatesZList, dim=0)
		if(fullRotationalInvariance):
			imageKeypointCoordinates = ATORpt_PTkeypoints.cropCoordinatesArray(imageKeypointCoordinates, ATORmaxNumberOfPolys//snapshotNumberOfKeypoints)
			imageKeypointCoordinates = ATORpt_PTkeypoints.generateKeypointPermutations(imageKeypointCoordinates)
		else:
			imageKeypointCoordinates = ATORpt_PTkeypoints.cropCoordinatesArray(imageKeypointCoordinates, ATORmaxNumberOfPolys)
		snapshotPixelCoordinates, snapshotMeshCoordinates, snapshotMeshValues, snapshotMeshFaces, snapshotMeshPolyCoordinates = ATORpt_PTmesh.getSnapshotMeshCoordinates(imageKeypointCoordinates, imagePath)
		imageKeypointCoordinates, snapshotPixelCoordinates, snapshotMeshCoordinates, snapshotMeshValues, snapshotMeshFaces, snapshotMeshPolyCoordinates = ATORpt_PTkeypoints.padCoordinatesArrays(imageKeypointCoordinates, snapshotPixelCoordinates, snapshotMeshCoordinates, snapshotMeshValues, snapshotMeshFaces, snapshotMeshPolyCoordinates)	#pad coordinates to ATORmaxNumberOfPolys

		if(debugSnapshotRenderFullImage):
			drawImageWithKeypoints(imagePath, imageKeypointCoordinates)
	
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
	
	if(snapshotRenderer == "pytorch3D"):
		meshCoordinates = snapshotMeshCoordinates
	
	keypointCoordinates = ATORpt_PTkeypoints.reorderKeypoints(keypointCoordinates)
	transformedMeshCoordinates = ATORpt_PTgeometricHashing.performGeometricHashingParallel(keypointCoordinates, meshCoordinates, meshValues=snapshotMeshValues, meshFaces=snapshotMeshFaces)
	print("geoHashing complete")
	transformedPatches = ATORpt_PTrenderer.resamplePixelCoordinates(transformedMeshCoordinates, snapshotMeshValues, snapshotMeshFaces, renderViewportSize, renderImageSize, centreSnapshots=False)	#transformedSnapshotPixelCoordinates	#after debug; centreSnapshots=False
	print("transformedPatches.shape = ", transformedPatches.shape)
	numPatchesPerImage = transformedPatches.shape[0]//batchSize
	transformedPatches = pt.reshape(transformedPatches, (batchSize, numPatchesPerImage, transformedPatches.shape[1], transformedPatches.shape[2], transformedPatches.shape[3]))	#shape: batchSize*ATORmaxNumberOfPolys, H, W, C
	transformedPatches = pt.permute(transformedPatches, (0, 1, 4, 2, 3))	#shape: batchSize, ATORmaxNumberOfPolys, C, H, W
	print("transformedPatches.shape = ", transformedPatches.shape)
	
	return transformedPatches

def drawImageWithKeypoints(imagePath, keypointCoordinates):

	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = TF.to_pil_image(image)
	
	for polyIndex in range(keypointCoordinates.shape[0]):
		polyKeypointCoordinates = keypointCoordinates[polyIndex]
		#mark keypoints on image
		R = (255, 0, 0) #RGB format
		G = (0, 255, 0) #RGB format
		B = (0, 0, 255) #RGB format
		M = (255, 0, 255) #RGB format
		Y = (255, 255, 0) #RGB format
		C = (0, 255, 255) #RGB format
		image.putpixel((int(polyKeypointCoordinates[0, xAxisGeometricHashing]), int(polyKeypointCoordinates[0, yAxisGeometricHashing])), M)
		image.putpixel((int(polyKeypointCoordinates[1, xAxisGeometricHashing]), int(polyKeypointCoordinates[1, yAxisGeometricHashing])), Y)
		image.putpixel((int(polyKeypointCoordinates[2, xAxisGeometricHashing]), int(polyKeypointCoordinates[2, yAxisGeometricHashing])), C)
		
	snapshotPixelCoordinates, snapshotMeshCoordinates, snapshotMeshValues, snapshotMeshFaces, snapshotMeshPolyCoordinates = ATORpt_PTmesh.getImageMeshCoordinates(image)
	renderViewportSizeImage = (300, 300)
	renderImageSizeImage = 1000
	transformedPatches = ATORpt_PTrenderer.resamplePixelCoordinates(snapshotMeshCoordinates, snapshotMeshValues, snapshotMeshFaces, renderViewportSizeImage, renderImageSizeImage, centreSnapshots=True)
	
	
def getImageCV(imagePath, zoomIndex):
	print("imagePath = ", imagePath)
	zoom = ATORpt_PTfeatures.getZoomValue(zoomIndex)
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	imageWidth = image.shape[1] // zoom
	imageHeight = image.shape[0] // zoom
	image = cv2.resize(image, (imageWidth, imageHeight))
	return image


