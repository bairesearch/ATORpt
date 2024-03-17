"""ATORpt_PTmesh.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt PT mesh

"""

import torch as pt
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import cv2
import math

from ATORpt_globalDefs import *
import ATORpt_operations

def getSnapshotMeshCoordinates(keypointCoordinates, imagePath):
	#resample snapshot mesh coordinates wrt keypoints
	#FUTURE: this function needs to be upgraded to perform the image resampling in parallel
	
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = TF.to_pil_image(image)
	
	snapshotPixelCoordinatesList = []
	snapshotMeshCoordinatesList = []	#mesh representation is offset by 0.5 pixels, size+1	#vertices
	snapshotMeshValuesList = []	#RGB values for each pixel/mesh coordinate in mesh	#colors
	snapshotMeshFacesList = []
	snapshotMeshPolyCoordinatesList = []
		
	for polyIndex in range(keypointCoordinates.shape[0]):
		polyKeypointCoordinates = keypointCoordinates[polyIndex]
		xMin = math.floor(pt.min(polyKeypointCoordinates[:, xAxisGeometricHashing]))
		yMin = math.floor(pt.min(polyKeypointCoordinates[:, yAxisGeometricHashing]))
		xMax = math.ceil(pt.max(polyKeypointCoordinates[:, xAxisGeometricHashing]))
		yMax = math.ceil(pt.max(polyKeypointCoordinates[:, yAxisGeometricHashing]))
		x = int(xMax - xMin)
		y = int(yMax - yMin)
		xPadded = x * ATORpatchPadding
		yPadded = y * ATORpatchPadding
		if(debugSnapshotRenderCroppedImage):
			#ensure snapshot is square
			x = 12
			y = 12
			xPadded = x*ATORpatchPadding
			yPadded = y*ATORpatchPadding
		if(ATORpatchPadding > 1):
			xMinPadded = int(xMin-(x*(ATORpatchPadding-1)/2))
			yMinPadded = int(yMin-(y*(ATORpatchPadding-1)/2))
		else:
			xMinPadded = int(xMin)
			yMinPadded = int(yMin)
		
		if(debugSnapshotRender or debugSnapshotRenderFinal):
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
				
		if(debugSnapshotRenderCroppedImage):
			print("x = ", x)
			print("y = ", y)
			print("xPadded = ", xPadded)
			print("yPadded = ", yPadded)
			print("xMin = ", xMin)
			print("yMin = ", yMin)
			print("xMinPadded = ", xMinPadded)
			print("yMinPadded = ", yMinPadded)

			#debug without resize:
			xSnapshot = xPadded
			ySnapshot = yPadded
			xScaling = 1.0
			yScaling = 1.0
			if(snapshotRenderExpectColorsDefinedForVerticesNotFaces):
				croppedImage = TF.crop(image, yMinPadded, xMinPadded, yPadded+1, xPadded+1)
			else:
				croppedImage = TF.crop(image, yMinPadded, xMinPadded, yPadded, xPadded)
			print("croppedImage.size[width] = ", croppedImage.size[0])
			print("croppedImage.size[height] = ", croppedImage.size[1])
			resampledImage = ATORpt_operations.pil_to_tensor(croppedImage)
		else:
			croppedImage = TF.crop(image, yMinPadded, xMinPadded, yPadded, xPadded)
			xSnapshot = ATORpatchSizeIntermediary[xAxisGeometricHashing]	#normaliseSnapshotLength*ATORpatchPadding*ATORpatchUpscaling
			ySnapshot = ATORpatchSizeIntermediary[yAxisGeometricHashing]	#normaliseSnapshotLength*ATORpatchPadding*ATORpatchUpscaling
			xScaling = xPadded/xSnapshot
			yScaling = yPadded/ySnapshot
			#print("xScaling = ", xScaling)
			#print("yScaling = ", yScaling)
			if(snapshotRenderExpectColorsDefinedForVerticesNotFaces):
				resampledImage = TF.resize(croppedImage, (ySnapshot+1, xSnapshot+1))
			else:
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

	'''
	print("snapshotPixelCoordinates.shape = ", snapshotPixelCoordinates.shape)
	print("snapshotMeshCoordinates.shape = ", snapshotMeshCoordinates.shape)
	print("snapshotMeshValues.shape = ", snapshotMeshValues.shape)
	print("snapshotMeshFaces.shape = ", snapshotMeshFaces.shape)
	print("snapshotMeshFaces.shape = ", snapshotMeshPolyCoordinates.shape)
	'''
	
	return snapshotPixelCoordinates, snapshotMeshCoordinates, snapshotMeshValues, snapshotMeshFaces, snapshotMeshPolyCoordinates
	
def generatePixelCoordinates(x, y, xScaling, yScaling, xMin, yMin):
	resampledImageShape = (y, x)
	polyPixelCoordinatesX = pt.arange(resampledImageShape[1]).expand(resampledImageShape[0], -1)	
	polyPixelCoordinatesY = pt.arange(resampledImageShape[0]).unsqueeze(1).expand(-1, resampledImageShape[1])
	#print("polyPixelCoordinatesX = ", polyPixelCoordinatesX)
	#print("polyPixelCoordinatesY = ", polyPixelCoordinatesY)
	polyPixelCoordinates = pt.stack((polyPixelCoordinatesX, polyPixelCoordinatesY), dim=-1)
	polyPixelCoordinates = pt.reshape(polyPixelCoordinates, (polyPixelCoordinates.shape[0]*polyPixelCoordinates.shape[1], polyPixelCoordinates.shape[2]))
	#convert polyPixelCoordinates back into original image coordinates frame:
	polyPixelCoordinates[:, xAxisGeometricHashing] = polyPixelCoordinates[:, xAxisGeometricHashing] * xScaling
	polyPixelCoordinates[:, yAxisGeometricHashing] = polyPixelCoordinates[:, yAxisGeometricHashing] * yScaling
	polyPixelCoordinates[:, xAxisGeometricHashing] = polyPixelCoordinates[:, xAxisGeometricHashing] + xMin
	polyPixelCoordinates[:, yAxisGeometricHashing] = polyPixelCoordinates[:, yAxisGeometricHashing] + yMin
	return polyPixelCoordinates
	
def generatePixelValues(resampledImage, snapshotPixelCoordinates, xSnapshot, ySnapshot, xMesh, yMesh, fullImage=False):
	numberOfVertices = snapshotPixelCoordinates.shape[0]
	if(fullImage):
		if(snapshotRenderExpectColorsDefinedForVerticesNotFaces):
			p2d = (0, 1, 0, 1) # pad last and second last dim by (0, 1)
			resampledImage = F.pad(resampledImage, p2d, "constant", snapshotRenderExpectColorsDefinedForVerticesNotFacesPadVal)
	#print("resampledImage.shape = ", resampledImage.shape)
	#resampledImage = resampledImage.permute(2, 1, 0)	#reshape from c, y, x (cv2/TF/Pil) to x, y, c
	resampledImage = resampledImage.permute(1, 2, 0)	#reshape from c, y, x (cv2/TF/Pil) to y, x, c
	snapshotMeshValues = pt.reshape(resampledImage, (resampledImage.shape[0]*resampledImage.shape[1], resampledImage.shape[2]))	
	polyMeshCoordinateIndices = pt.arange(xMesh * yMesh).reshape(xMesh, yMesh)
	if(snapshotRenderTris):
		#consider swapping order of tri vertices (interpreted as direction of face wrt lighting)
		snapshotMeshFaces1a = polyMeshCoordinateIndices[0:xSnapshot, 0:ySnapshot]
		snapshotMeshFaces2a = polyMeshCoordinateIndices[1:xSnapshot+1, 0:ySnapshot]
		snapshotMeshFaces3a = polyMeshCoordinateIndices[0:xSnapshot, 1:ySnapshot+1]
		snapshotMeshFaces1b = polyMeshCoordinateIndices[0:xSnapshot, 1:ySnapshot+1]
		snapshotMeshFaces2b = polyMeshCoordinateIndices[1:xSnapshot+1, 0:ySnapshot]
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

def getImageMeshCoordinates(image):

	image_size = image.size
	width, height = image_size
	
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
	snapshotMeshValues, snapshotMeshFaces = generatePixelValues(image, snapshotPixelCoordinates, xSnapshot, ySnapshot, xMesh, yMesh, fullImage=True) 
	#print("snapshotMeshCoordinates.shape = ", snapshotMeshCoordinates.shape)
	snapshotMeshPolyCoordinates = generateMeshPolyCoordinates(snapshotMeshCoordinates)

	snapshotPixelCoordinates = snapshotPixelCoordinates.unsqueeze(0)
	snapshotMeshCoordinates = snapshotMeshCoordinates.unsqueeze(0)
	snapshotMeshValues = snapshotMeshValues.unsqueeze(0)
	snapshotMeshFaces = snapshotMeshFaces.unsqueeze(0)
	snapshotMeshPolyCoordinates = snapshotMeshPolyCoordinates.unsqueeze(0)
	
	snapshotMeshCoordinates = centrePixelCoordinates(snapshotMeshCoordinates, xMesh, yMesh)
		
	return snapshotPixelCoordinates, snapshotMeshCoordinates, snapshotMeshValues, snapshotMeshFaces, snapshotMeshPolyCoordinates
		
def centrePixelCoordinates(snapshotPixelCoordinates, x, y):
	print("snapshotPixelCoordinates.shape = ", snapshotPixelCoordinates.shape)
	snapshotPixelCoordinates[:, :, xAxisGeometricHashing] = snapshotPixelCoordinates[:, :, xAxisGeometricHashing] - x/2
	snapshotPixelCoordinates[:, :, yAxisGeometricHashing] = snapshotPixelCoordinates[:, :, yAxisGeometricHashing] - y/2
	return snapshotPixelCoordinates
