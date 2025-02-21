"""ATORpt_PTmesh.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

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

from ATORpt_PTglobalDefs import *
import ATORpt_operations

def getSnapshotMeshCoordinates(use3DOD, keypointCoordinates, image, imageDepth=None):
	#resample snapshot mesh coordinates wrt keypoints
	#FUTURE: this function needs to be upgraded to perform the image resampling in parallel
	
	image = TF.to_pil_image(image)
	
	snapshotPixelCoordinatesList = []
	snapshotMeshCoordinatesList = []	#mesh representation is offset by 0.5 pixels, size+1	#vertices
	snapshotMeshValuesList = []	#RGB values for each pixel/mesh coordinate in mesh	#colors
	snapshotMeshFacesList = []
	snapshotMeshPolyCoordinatesList = []
	
	if(use3DOD):
		ATORpatchPadding = ATORpatchPadding3DOD
		ATORpatchSizeIntermediary = ATORpatchSizeIntermediary3DOD
	else:
		ATORpatchPadding = ATORpatchPadding2DOD
		ATORpatchSizeIntermediary =	ATORpatchSizeIntermediary2DOD
		
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
			xSqr = 12
			ySqr = 12
			xPadded = xSqr*ATORpatchPadding
			yPadded = ySqr*ATORpatchPadding
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
			
		if(not use3DOD or ATOR3DODgeoHashingScale):
			if(debugSnapshotRenderCroppedImage):
				#debug without resize:
				xSnapshot = xPadded
				ySnapshot = yPadded
				xScaling = 1.0
				yScaling = 1.0
				resampledImageVert, resampledImageDepthVert = crop(use3DOD, image, yMinPadded, xMinPadded, yPadded+1, xPadded+1, imageDepth)
				resampledImage, resampledImageDepth = crop(use3DOD, image, yMinPadded, xMinPadded, yPadded, xPadded, imageDepth)
			else:
				croppedImage, croppedImageDepth = crop(use3DOD, image, yMinPadded, xMinPadded, yPadded, xPadded, imageDepth)
				xSnapshot = ATORpatchSizeIntermediary[xAxisGeometricHashing]	#normaliseSnapshotLength*ATORpatchPadding*ATORpatchUpscaling
				ySnapshot = ATORpatchSizeIntermediary[yAxisGeometricHashing]	#normaliseSnapshotLength*ATORpatchPadding*ATORpatchUpscaling
				xScaling = xPadded/xSnapshot
				yScaling = yPadded/ySnapshot
				resampledImageVert, resampledImageDepthVert = resize(use3DOD, croppedImage, ySnapshot+1, xSnapshot+1, croppedImageDepth)
				resampledImage, resampledImageDepth = resize(use3DOD, croppedImage, ySnapshot, xSnapshot, croppedImageDepth)
		else:
			x = min(image.width, image.height)	#ensure square
			y = min(image.width, image.height)	#ensure square
			resampledImageVert, resampledImageDepthVert = crop(use3DOD, image, 0, 0, y+1, x+1, imageDepth)
			resampledImage, resampledImageDepth = crop(use3DOD, image, 0, 0, y, x, imageDepth)
			#print("resampledImageVert.size = ", resampledImageVert.size)
			#print("resampledImage.size = ", resampledImage.size)
			#print("resampledImageDepthVert.shape = ", resampledImageDepthVert.shape)
			#print("resampledImageDepth.shape = ", resampledImageDepth.shape)
			xSnapshot = x
			ySnapshot = y
			xScaling = 1.0
			yScaling = 1.0
		if(snapshotRenderExpectColorsDefinedForVerticesNotFaces):
			resampledImage = resampledImageVert
		resampledImage = convertImageToTensor(resampledImage)
	
		xMesh = xSnapshot+1
		yMesh = ySnapshot+1
		#print("xSnapshot = ", xSnapshot)
		#print("ySnapshot = ", ySnapshot)
		snapshotPixelCoordinates = generatePixelCoordinates(use3DOD, xSnapshot, ySnapshot, xScaling, yScaling, xMinPadded, yMinPadded, resampledImageDepth)
		snapshotMeshCoordinates = generatePixelCoordinates(use3DOD, xMesh, yMesh, xScaling, yScaling, xMinPadded, yMinPadded, resampledImageDepthVert)
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
	
def generatePixelCoordinates(use3DOD, x, y, xScaling, yScaling, xMin, yMin, imageDepth=None):
	resampledImageShape = (y, x)
	polyPixelCoordinatesX = pt.arange(resampledImageShape[1]).expand(resampledImageShape[0], -1)	
	polyPixelCoordinatesY = pt.arange(resampledImageShape[0]).unsqueeze(1).expand(-1, resampledImageShape[1])
	polyPixelCoordinates = pt.stack((polyPixelCoordinatesX, polyPixelCoordinatesY), dim=-1)
	polyPixelCoordinates = pt.reshape(polyPixelCoordinates, (polyPixelCoordinates.shape[0]*polyPixelCoordinates.shape[1], polyPixelCoordinates.shape[2]))
	if(use3DOD):
		polyPixelCoordinatesZ = pt.reshape(imageDepth, (imageDepth.shape[0]*imageDepth.shape[1], 1))
		#print("polyPixelCoordinates.shape = ", polyPixelCoordinates.shape)
		#print("polyPixelCoordinatesZ.shape = ", polyPixelCoordinatesZ.shape)
		polyPixelCoordinates = pt.cat((polyPixelCoordinates, polyPixelCoordinatesZ), dim=1)	#add Z dimension to coordinates	#zAxisGeometricHashing
	#convert polyPixelCoordinates back into original image coordinates frame:
	polyPixelCoordinates[:, xAxisGeometricHashing] = polyPixelCoordinates[:, xAxisGeometricHashing] * xScaling
	polyPixelCoordinates[:, yAxisGeometricHashing] = polyPixelCoordinates[:, yAxisGeometricHashing] * yScaling
	polyPixelCoordinates[:, xAxisGeometricHashing] = polyPixelCoordinates[:, xAxisGeometricHashing] + xMin
	polyPixelCoordinates[:, yAxisGeometricHashing] = polyPixelCoordinates[:, yAxisGeometricHashing] + yMin
	return polyPixelCoordinates
	
def generatePixelValues(resampledImage, snapshotPixelCoordinates, xSnapshot, ySnapshot, xMesh, yMesh):
	numberOfVertices = snapshotPixelCoordinates.shape[0]
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
	#CHECKTHIS: supports use3DOD
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

def getImageMeshCoordinates(use3DOD, image, imageDepth=None):

	image_size = image.size
	width, height = image_size
	
	xMin = 200
	yMin = 200
	xSnapshot = 300
	ySnapshot = 300
	print("height = ", height)

	imageVert, imageDepthVert = crop(use3DOD, image, yMin, xMin, ySnapshot+1, xSnapshot+1, imageDepth)	#make sure image is square
	image, imageDepth = crop(use3DOD, image, yMin, xMin, ySnapshot, xSnapshot, imageDepth)	#make sure image is square
	if(snapshotRenderExpectColorsDefinedForVerticesNotFaces):
		image = imageVert
	image = convertImageToTensor(image)
	
	print("image.shape = ", image.shape)
	
	xMin = 0	#set start coordinates 0 zero for immediate rendering (without geohashing)
	yMin = 0 	#set start coordinates 0 zero for immediate rendering (without geohashing)
	xMesh = xSnapshot+1
	yMesh = ySnapshot+1	
	xScaling = 1.0
	yScaling = 1.0
	snapshotPixelCoordinates = generatePixelCoordinates(use3DOD, xSnapshot, ySnapshot, xScaling, yScaling, xMin, yMin, imageDepth)
	snapshotMeshCoordinates = generatePixelCoordinates(use3DOD, xMesh, yMesh, xScaling, yScaling, xMin, yMin, imageDepthVert)
	snapshotMeshValues, snapshotMeshFaces = generatePixelValues(image, snapshotPixelCoordinates, xSnapshot, ySnapshot, xMesh, yMesh) 
	#print("snapshotMeshCoordinates.shape = ", snapshotMeshCoordinates.shape)
	snapshotMeshPolyCoordinates = generateMeshPolyCoordinates(snapshotMeshCoordinates)

	snapshotPixelCoordinates = snapshotPixelCoordinates.unsqueeze(0)
	snapshotMeshCoordinates = snapshotMeshCoordinates.unsqueeze(0)
	snapshotMeshValues = snapshotMeshValues.unsqueeze(0)
	snapshotMeshFaces = snapshotMeshFaces.unsqueeze(0)
	snapshotMeshPolyCoordinates = snapshotMeshPolyCoordinates.unsqueeze(0)
	
	snapshotMeshCoordinates = centrePixelCoordinates(snapshotMeshCoordinates, xMesh, yMesh)
		
	return snapshotPixelCoordinates, snapshotMeshCoordinates, snapshotMeshValues, snapshotMeshFaces, snapshotMeshPolyCoordinates

def convertImageToTensor(image):
	resampledImage = ATORpt_operations.pil_to_tensor(image).to(devicePreprocessing)
	return resampledImage
	
def centrePixelCoordinates(snapshotPixelCoordinates, x, y):
	print("snapshotPixelCoordinates.shape = ", snapshotPixelCoordinates.shape)
	snapshotPixelCoordinates[:, :, xAxisGeometricHashing] = snapshotPixelCoordinates[:, :, xAxisGeometricHashing] - x/2
	snapshotPixelCoordinates[:, :, yAxisGeometricHashing] = snapshotPixelCoordinates[:, :, yAxisGeometricHashing] - y/2
	return snapshotPixelCoordinates

def crop(use3DOD, image, yMin, xMin, y, x, imageDepth=None):
	croppedImage = TF.crop(image, yMin, xMin, y, x)
	if(use3DOD):
		croppedImageDepth = cropImageDepth(imageDepth, yMin, xMin, y, x)
	else:
		croppedImageDepth = None
	return croppedImage, croppedImageDepth
	
def resize(use3DOD, image, yNewSize, xNewSize, imageDepth=None):
	#print("image = ", image)
	#print("yNewSize = ", yNewSize)
	#print("xNewSize = ", xNewSize)
	resampledImage = TF.resize(image, (yNewSize, xNewSize))
	if(use3DOD):
		resampledImageDepth = resizeImageDepth(imageDepth, yNewSize, xNewSize)
	else:
		resampledImageDepth = None
	return resampledImage, resampledImageDepth

def cropImageDepth(imageDepth, yMin, xMin, y, x):
	imageDepth = pt.permute(imageDepth, (1, 0))	#switch from W,H to H,W
	imageDepth = crop_and_pad(imageDepth, yMin, xMin, y, x)
	imageDepth = pt.permute(imageDepth, (0, 1))	#switch from H,W to W,H
	return imageDepth

def resizeImageDepth(imageDepth, new_height, new_width):
	original_height, original_width = imageDepth.shape
	height_scale = original_height / new_height
	width_scale = original_width / new_width
	new_y, new_x = pt.meshgrid(pt.arange(new_height, device=devicePreprocessing), pt.arange(new_width, device=devicePreprocessing), indexing='ij')
	y = (new_y.float() + 0.5) * height_scale - 0.5
	x = (new_x.float() + 0.5) * width_scale - 0.5
	y_nearest = pt.clamp(y.round().long(), 0, original_height - 1)
	x_nearest = pt.clamp(x.round().long(), 0, original_width - 1)
	imageDepth = imageDepth[y_nearest, x_nearest]
	return imageDepth
	
def crop_and_pad(tensor, top, left, height, width):
	"""
	Crops a 2D or 3D tensor to a specified height and width, padding with zeros if necessary.

	Parameters:
	- tensor (torch.Tensor): The input tensor. Expected shape: (C, H, W) or (H, W).
	- top (int): The top coordinate for the crop.
	- left (int): The left coordinate for the crop.
	- height (int): The height of the crop.
	- width (int): The width of the crop.

	Returns:
	- torch.Tensor: The cropped and potentially zero-padded tensor.
	"""
	# Ensure tensor is at least 3D
	was_2d = False
	if tensor.dim() == 2:
		tensor = tensor.unsqueeze(0)  # Add a channel dimension if it's missing
		was_2d = True

	_, orig_h, orig_w = tensor.size()

	# Calculate padding requirements
	pad_left = max(0, -left)
	pad_top = max(0, -top)
	pad_right = max(0, left + width - orig_w)
	pad_bottom = max(0, top + height - orig_h)

	# Apply padding if needed
	if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
		tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), 'constant', ATORpatchCropPaddingValue)

	# Adjust crop coordinates to the new padded tensor dimensions
	crop_left = max(0, left)
	crop_top = max(0, top)

	# Crop the tensor
	cropped_tensor = tensor[:, crop_top:crop_top + height, crop_left:crop_left + width]

	# If the original tensor was 2D, remove the added channel dimension
	if was_2d:
		cropped_tensor = cropped_tensor.squeeze(0)

	return cropped_tensor

