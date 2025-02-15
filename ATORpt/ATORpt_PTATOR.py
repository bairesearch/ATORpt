"""ATORpt_PTATOR.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

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
import ATORpt_PTgeometricHashing2DOD
import ATORpt_PTgeometricHashing3DOD
import ATORpt_operations
import ATORpt_PTrenderer
import ATORpt_PTdepth3DOD

def generateATORpatches(use3DOD, imagePaths, train):
	#image polys (ie polys across all images) are merged into single dim for geometric hashing and rendering

	keypointCoordinatesList = []
	snapshotPixelCoordinatesList = []
	snapshotMeshCoordinatesList = []
	snapshotMeshValuesList = []
	snapshotMeshFacesList = []
	snapshotMeshPolyCoordinatesList = []
	
	for imageIndex, imagePath in enumerate(imagePaths):
		if(debugVerbose):
			print("imageIndex = ", imageIndex)
		imageKeypointCoordinatesZoomList = []
		for zoomIndex in range(numberOfZoomLevels):
			if(debugVerbose):
				print("\tzoomIndex = ", zoomIndex)
			image, imageDepth = getImage(use3DOD, imagePath, applyZoom=True, zoomIndex=zoomIndex)
			imageFeatureCoordinatesZoom = ATORpt_PTfeatures.featureDetection(image, zoomIndex)
			imageKeypointCoordinatesZoom = ATORpt_PTkeypoints.performKeypointDetection(imageFeatureCoordinatesZoom)
			imageKeypointCoordinatesZoomList.append(imageKeypointCoordinatesZoom)
		imageKeypointCoordinates = pt.cat(imageKeypointCoordinatesZoomList, dim=0)
		if(fullRotationalInvariance):
			imageKeypointCoordinates = ATORpt_PTkeypoints.cropCoordinatesArray(imageKeypointCoordinates, ATORmaxNumberOfPolys//snapshotNumberOfKeypoints)
			imageKeypointCoordinates = ATORpt_PTkeypoints.generateKeypointPermutations(imageKeypointCoordinates)
		else:
			imageKeypointCoordinates = ATORpt_PTkeypoints.cropCoordinatesArray(imageKeypointCoordinates, ATORmaxNumberOfPolys)
		if(numberOfZoomLevels > 1):
			image, imageDepth = getImage(use3DOD, imagePath)
		snapshotPixelCoordinates, snapshotMeshCoordinates, snapshotMeshValues, snapshotMeshFaces, snapshotMeshPolyCoordinates = ATORpt_PTmesh.getSnapshotMeshCoordinates(use3DOD, imageKeypointCoordinates, image, imageDepth)
		imageKeypointCoordinates, snapshotPixelCoordinates, snapshotMeshCoordinates, snapshotMeshValues, snapshotMeshFaces, snapshotMeshPolyCoordinates = ATORpt_PTkeypoints.padCoordinatesArrays(imageKeypointCoordinates, snapshotPixelCoordinates, snapshotMeshCoordinates, snapshotMeshValues, snapshotMeshFaces, snapshotMeshPolyCoordinates)	#pad coordinates to ATORmaxNumberOfPolys

		if(debugSnapshotRenderFullImage):
			drawImageWithKeypoints(use3DOD, image, imageKeypointCoordinates, imageDepth)
	
		#print("keypointCoordinates.shape = ", keypointCoordinates.shape)
		imageKeypointCoordinates = ATORpt_PTkeypoints.reorderKeypoints(imageKeypointCoordinates)
		if(use3DOD):
			imageKeypointCoordinates = ATORpt_PTkeypoints.addZcoordinates(imageKeypointCoordinates, imageDepth)
		
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
		#snapshotMeshPolyCoordinates is not used by pytorch3D
	
	if(debugVerbose):
		print("geoHashing start:")
	if(use3DOD):
		transformedMeshCoordinates = ATORpt_PTgeometricHashing3DOD.performGeometricHashingParallel(keypointCoordinates, meshCoordinates, meshValues=snapshotMeshValues, meshFaces=snapshotMeshFaces)
	else:
		transformedMeshCoordinates = ATORpt_PTgeometricHashing2DOD.performGeometricHashingParallel(keypointCoordinates, meshCoordinates, meshValues=snapshotMeshValues, meshFaces=snapshotMeshFaces)
	if(use3DOD):
		renderViewportSize = renderViewportSize3DOD
	else:
		renderViewportSize = renderViewportSize2DOD
	if(debugVerbose):
		print("geoHashing complete")
	transformedPatches = ATORpt_PTrenderer.resamplePixelCoordinates(use3DOD, transformedMeshCoordinates, snapshotMeshValues, snapshotMeshFaces, renderViewportSize, renderImageSize, centreSnapshots=True)	#transformedSnapshotPixelCoordinates	#after debug; centreSnapshots=False
	#print("transformedPatches.shape = ", transformedPatches.shape)
	numPatchesPerImage = transformedPatches.shape[0]//batchSize
	transformedPatches = pt.reshape(transformedPatches, (batchSize, numPatchesPerImage, transformedPatches.shape[1], transformedPatches.shape[2], transformedPatches.shape[3]))	#shape: batchSize*ATORmaxNumberOfPolys, H, W, C
	transformedPatches = pt.permute(transformedPatches, (0, 1, 4, 2, 3))	#shape: batchSize, ATORmaxNumberOfPolys, C, H, W
	#print("transformedPatches.shape = ", transformedPatches.shape)
	
	return transformedPatches

def drawImageWithKeypoints(use3DOD, image, keypointCoordinates, imageDepth=None):
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
		
	snapshotPixelCoordinates, snapshotMeshCoordinates, snapshotMeshValues, snapshotMeshFaces, snapshotMeshPolyCoordinates = ATORpt_PTmesh.getImageMeshCoordinates(use3DOD, image, imageDepth)
	renderViewportSizeImage = (300, 300)
	renderImageSizeImage = 1000
	transformedPatches = ATORpt_PTrenderer.resamplePixelCoordinates(use3DOD, snapshotMeshCoordinates, snapshotMeshValues, snapshotMeshFaces, renderViewportSizeImage, renderImageSizeImage, centreSnapshots=True)
	
def getImage(use3DOD, imagePath, applyZoom=False, zoomIndex=-1):
	imageDepth = None
	if(use3DOD):
		if(generate3DODfrom2DOD):
			image = getImageCV(imagePath)
			imageDepth = ATORpt_PTdepth3DOD.deriveImageDepth(image)
		elif(generate3DODfromParallax):
			image, imageDepth = ATORpt_PTdepth3DOD.getImage3D(imagePath)	
			imageDepth = ATORpt_PTmesh.convertImageToTensor(imageDepth)	#all imageDepth operations (resize, crop etc) assume tensor format
			imageDepth = imageDepth.squeeze(0)
	else:
		image = getImageCV(imagePath)
	if(applyZoom):
		zoom = ATORpt_PTfeatures.getZoomValue(zoomIndex)
		imageWidth = image.shape[1] // zoom
		imageHeight = image.shape[0] // zoom
		image = cv2.resize(image, (imageWidth, imageHeight))
		if(use3DOD):
			imageDepth = ATORpt_PTmesh.resizeImageDepth(imageDepth, imageHeight, imageWidth)
	return image, imageDepth
				
def getImageCV(imagePath):
	#shape = [Height, Width, Channels] where Channels is [Blue, Green, Red] (opencv standard)
	if(debugVerbose):
		print("imagePath = ", imagePath)
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	return image
	
def deriveImageDepth(image):
	incomplete
	return imageDepth
	
def getImage3D(imagePath):
	incomplete
	#assume imagePath contains parallax derived pixel+depthMap
	return image, imageDepth
