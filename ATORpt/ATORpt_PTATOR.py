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
import torchvision.transforms.functional as TF
from PIL import Image
import cv2

from ATORpt_globalDefs import *
import ATORpt_PTfeatureDetector
import ATORpt_PTpolyKeypointGenerator
import ATORpt_PTgeometricHashing
import ATORpt_operations

def generateATORpatches(imagePaths, train):

	keypointCoordinatesList = []
	snapshotMeshCoordinatesList = []
	snapshotPixelCoordinatesList = []
	snapshotPixelValuesList = []
	snapshotPixelFacesList = []
	
	for imageIndex, imagePath in enumerate(imagePaths):
		imageKeypointCoordinatesZList = []
		for zoomIndex in range(numberOfZoomLevels):
			image = getImageCV(imagePath, zoomIndex)	#shape = [Height, Width, Channels] where Channels is [Blue, Green, Red] (opencv standard)
			imageFeatureCoordinatesZ = ATORpt_PTfeatureDetector.featureDetection(image, zoomIndex)
			imageKeypointCoordinatesZ = ATORpt_PTpolyKeypointGenerator.performKeypointDetection(imageFeatureCoordinatesZ)
			imageKeypointCoordinatesZList.append(imageKeypointCoordinatesZ)
		imageKeypointCoordinates = pt.cat(imageKeypointCoordinatesZList, dim=0)
		snapshotMeshCoordinates, snapshotPixelCoordinates, snapshotPixelValues, snapshotPixelFaces = getSnapshotMeshCoordinates(imageKeypointCoordinates, imagePath)
		imageKeypointCoordinates, snapshotMeshCoordinates, snapshotPixelCoordinates, snapshotPixelValues, snapshotPixelFaces = ATORpt_PTpolyKeypointGenerator.padCoordinatesArrays(imageKeypointCoordinates, snapshotMeshCoordinates, snapshotPixelCoordinates, snapshotPixelValues, snapshotPixelFaces)
		
		keypointCoordinatesList.append(imageKeypointCoordinates)
		snapshotMeshCoordinatesList.append(snapshotMeshCoordinates)
		snapshotPixelCoordinatesList.append(snapshotPixelCoordinates)
		snapshotPixelValuesList.append(snapshotPixelValues)
		snapshotPixelFacesList.append(snapshotPixelFaces)	
		
	keypointCoordinates = pt.cat(keypointCoordinatesList, dim=0)
	snapshotMeshCoordinates = pt.cat(snapshotMeshCoordinatesList, dim=0)
	snapshotPixelCoordinates = pt.cat(snapshotPixelCoordinatesList, dim=0)
	snapshotPixelValues = pt.cat(snapshotPixelValuesList, dim=0)
	snapshotPixelFaces = pt.cat(snapshotPixelFacesList, dim=0)
	
	#print("snapshotMeshCoordinates.shape = ", snapshotMeshCoordinates.shape)
	#print("snapshotPixelCoordinates.shape = ", snapshotPixelCoordinates.shape)
	#print("snapshotPixelValues.shape = ", snapshotPixelValues.shape)
	#print("snapshotPixelFaces.shape = ", snapshotPixelFaces.shape)

	transformedSnapshotPixelCoordinates = ATORpt_PTgeometricHashing.performGeometricHashingParallel(keypointCoordinates, snapshotPixelCoordinates, pixelValues=snapshotPixelValues)	#after debug: snapshotMeshCoordinates
	transformedPatches = resamplePixelCoordinates(transformedSnapshotPixelCoordinates, snapshotPixelValues, snapshotPixelFaces)
	return transformedPatches

def getImageTF(imagePath):
	image = Image.open(imagePath)
	#image = TF.to_pil_image(image)
	return image

def getImageCV(imagePath, zoomIndex):
	zoom = ATORpt_PTfeatureDetector.getZoomValue(zoomIndex)
	image = cv2.imread(imagePath)
	imageWidth = image.shape[1] // zoom
	imageHeight = image.shape[0] // zoom
	image = cv2.resize(image, (imageWidth, imageHeight))
	return image

def getSnapshotMeshCoordinates(keypointCoordinates, imagePath):
	#resample snapshot mesh coordinates wrt keypoints
	#FUTURE: this function needs to be upgraded to perform the image resampling in parallel
	
	image = cv2.imread(imagePath)
	image = TF.to_pil_image(image)
	
	snapshotMeshCoordinatesList = []	#mesh representation is offset by 0.5 pixels, size+1	#vertices
	snapshotPixelCoordinatesList = []	
	snapshotPixelValuesList = []	#RGB values for each pixel/mesh coordinate in mesh	#colors
	snapshotPixelFacesList = []
	
	print("keypointCoordinates.shape = ", keypointCoordinates.shape)
	
	for polyIndex in range(keypointCoordinates.shape[0]):
		polyKeypointCoordinates = keypointCoordinates[polyIndex]
		xMin = pt.min(polyKeypointCoordinates[:, 0])
		yMin = pt.min(polyKeypointCoordinates[:, 1])
		xMax = pt.max(polyKeypointCoordinates[:, 0])
		yMax = pt.max(polyKeypointCoordinates[:, 1])
		x = int((xMax - xMin) * ATORpatchPadding)
		y = int((yMax - yMin) * ATORpatchPadding)
		xMinPadded = int(xMin-(x*ATORpatchPadding//2))
		yMinPadded = int(yMin-(y*ATORpatchPadding//2))
		croppedImage = TF.crop(image, yMinPadded, xMinPadded, y, x)
		xSnapshot = normaliseSnapshotLength*ATORpatchPadding*ATORpatchUpscaling
		ySnapshot = normaliseSnapshotLength*ATORpatchPadding*ATORpatchUpscaling
		xScaling = x/xSnapshot
		yScaling = y/ySnapshot
		resampledImage = TF.resize(croppedImage, (ySnapshot, xSnapshot))
		#resampledImage = transformToTensor(resampledImage)
		resampledImage = ATORpt_operations.pil_to_tensor(resampledImage)
		#print("resampledImage.shape = ", resampledImage.shape)
		
		xMesh = xSnapshot+1
		yMesh = ySnapshot+1
		snapshotMeshCoordinates = generatePixelCoordinates(xMesh, yMesh, xScaling, yScaling, xMinPadded, yMinPadded)
		snapshotPixelCoordinates = generatePixelCoordinates(xSnapshot, ySnapshot, xScaling, yScaling, xMinPadded, yMinPadded)
		#print("resampledImage.shape = ", resampledImage.shape)
		#print("snapshotPixelCoordinates.shape = ", snapshotPixelCoordinates.shape)
		snapshotPixelValues, snapshotPixelFaces = generatePixelValues(resampledImage, snapshotPixelCoordinates, xSnapshot, ySnapshot, xMesh, yMesh) 
	
		#print("snapshotPixelCoordinates.shape = ", snapshotPixelCoordinates.shape)
		#print("snapshotPixelValues.shape = ", snapshotPixelValues.shape)
		#print("snapshotPixelFaces.shape = ", snapshotPixelFaces.shape)
		
		snapshotMeshCoordinatesList.append(snapshotMeshCoordinates)
		snapshotPixelCoordinatesList.append(snapshotPixelCoordinates)
		snapshotPixelValuesList.append(snapshotPixelValues)
		snapshotPixelFacesList.append(snapshotPixelFaces)
	
	snapshotMeshCoordinates = pt.stack(snapshotMeshCoordinatesList, dim=0).to(device)
	snapshotPixelCoordinates = pt.stack(snapshotPixelCoordinatesList, dim=0).to(device)
	snapshotPixelValues = pt.stack(snapshotPixelValuesList, dim=0).to(device)
	snapshotPixelFaces = pt.stack(snapshotPixelValuesList, dim=0).to(device)

	#print("snapshotMeshCoordinates.shape = ", snapshotMeshCoordinates.shape)
	#print("snapshotPixelCoordinates.shape = ", snapshotPixelCoordinates.shape)
	#print("snapshotPixelValues.shape = ", snapshotPixelValues.shape)
	#print("snapshotPixelFaces.shape = ", snapshotPixelFaces.shape)

	return snapshotMeshCoordinates, snapshotPixelCoordinates, snapshotPixelValues, snapshotPixelFaces
	
def generatePixelCoordinates(x, y, xScaling, yScaling, xMinPadded, yMinPadded):
	resampledImageShape = (y, x)
	polyPixelCoordinatesY = pt.arange(resampledImageShape[1]).expand(resampledImageShape[0], -1)
	polyPixelCoordinatesX = pt.arange(resampledImageShape[0]).unsqueeze(1).expand(-1, resampledImageShape[1])
	#print("polyPixelCoordinatesY = ", polyPixelCoordinatesY.shape)
	#print("polyPixelCoordinatesX = ", polyPixelCoordinatesX.shape)
	polyPixelCoordinates = pt.stack((polyPixelCoordinatesY, polyPixelCoordinatesX), dim=-1)
	polyPixelCoordinates[:, 0] = polyPixelCoordinates[:, 0] * yScaling
	polyPixelCoordinates[:, 1] = polyPixelCoordinates[:, 1] * xScaling
	polyPixelCoordinates[:, 0] = polyPixelCoordinates[:, 0] + yMinPadded
	polyPixelCoordinates[:, 1] = polyPixelCoordinates[:, 1] + xMinPadded
	polyPixelCoordinates = pt.reshape(polyPixelCoordinates, (polyPixelCoordinates.shape[0]*polyPixelCoordinates.shape[1], polyPixelCoordinates.shape[2]))
	#print("polyPixelCoordinates = ", polyPixelCoordinates.shape)
	return polyPixelCoordinates
	
def generatePixelValues(resampledImage, snapshotPixelCoordinates, xSnapshot, ySnapshot, xMesh, yMesh):
	numberOfVertices = snapshotPixelCoordinates.shape[0]
	#print("resampledImage.shape = ", resampledImage.shape)
	snapshotPixelValues = pt.reshape(resampledImage, (resampledImage.shape[0], resampledImage.shape[1]*resampledImage.shape[2]))
	#print("snapshotPixelValues.shape = ", snapshotPixelValues.shape)
	snapshotPixelFaces1 = pt.arange(0, ySnapshot*xMesh-1)
	snapshotPixelFaces2 = pt.arange(1, ySnapshot*xMesh)
	snapshotPixelFaces3 = pt.arange(xMesh, yMesh*xMesh-1)
	snapshotPixelFaces4 = pt.arange(xMesh+1, yMesh*xMesh)
	#print("snapshotPixelFaces1.shape = ", snapshotPixelFaces1.shape)
	snapshotPixelFaces = pt.stack((snapshotPixelFaces1, snapshotPixelFaces2, snapshotPixelFaces3, snapshotPixelFaces4), dim=1)
	#print("snapshotPixelFaces.shape = ", snapshotPixelFaces.shape)
	return snapshotPixelValues, snapshotPixelFaces

def resamplePixelCoordinates(transformedSnapshotPixelCoordinates, snapshotPixelValues, snapshotPixelFaces):
	image_size = normaliseSnapshotLength*ATORpatchPadding
	transformedPatches = rasterize_meshes(transformedSnapshotPixelCoordinates, snapshotPixelFaces, snapshotPixelValues, image_size)
	return transformedPatches
		
def rasterize_meshes(vertices_batch, faces_batch, face_colors_batch, image_size):
	#CHECKTHIS
	
    # Create an empty batch of image tensors
    batch_size = vertices_batch.shape[0]
    image = pt.zeros(batch_size, 3, image_size, image_size, dtype=pt.float32)
    
    # Unpack face_colors into RGB channels
    red = face_colors_batch[:, :, 0].unsqueeze(2)
    green = face_colors_batch[:, :, 1].unsqueeze(2)
    blue = face_colors_batch[:, :, 2].unsqueeze(2)
    
    # Normalize vertex coordinates to image space
    vertices = (vertices_batch + 1) * (image_size - 1) / 2
    
    # Extract vertices of each face
    vertices_per_face = vertices_batch[:, faces_batch]
    
    # Calculate edge vectors and triangle area
    AB = vertices_per_face[:, :, 1] - vertices_per_face[:, :, 0]
    BC = vertices_per_face[:, :, 2] - vertices_per_face[:, :, 1]
    CD = vertices_per_face[:, :, 3] - vertices_per_face[:, :, 2]
    DA = vertices_per_face[:, :, 0] - vertices_per_face[:, :, 3]

    cross_product1 = pt.cross(AB, BC)
    cross_product2 = pt.cross(BC, CD)
    cross_product3 = pt.cross(CD, DA)
    cross_product4 = pt.cross(DA, AB)
    
    twice_triangle_area1 = cross_product1[:, :, :, 2].unsqueeze(3)
    twice_triangle_area2 = cross_product2[:, :, :, 2].unsqueeze(3)
    twice_triangle_area3 = cross_product3[:, :, :, 2].unsqueeze(3)
    twice_triangle_area4 = cross_product4[:, :, :, 2].unsqueeze(3)
    
    # Calculate barycentric coordinates of each pixel in the image
    pixel_coords = pt.stack(pt.meshgrid(pt.arange(image_size), pt.arange(image_size)), dim=2)
    pixel_coords = pixel_coords.to(vertices.device).float()
    pixel_coords = pixel_coords.unsqueeze(0).expand(batch_size, -1, -1, -1)
    pixel_coords[..., 0] += 0.5
    pixel_coords[..., 1] += 0.5
    
    v0 = vertices_per_face[:, :, 0]
    v0_x, v0_y = v0[:, :, 0], v0[:, :, 1]
    pixel_coords = pixel_coords - v0.unsqueeze(-2).unsqueeze(-2)
    
    pixel_coords = pixel_coords.permute(0, 3, 1, 2)
    
    w1 = (pixel_coords[..., 0] * BC[..., 0] - pixel_coords[..., 1] * AB[..., 0]) / twice_triangle_area1
    w2 = (pixel_coords[..., 0] * CD[..., 0] - pixel_coords[..., 1] * BC[..., 0]) / twice_triangle_area2
    w3 = (pixel_coords[..., 0] * DA[..., 0] - pixel_coords[..., 1] * CD[..., 0]) / twice_triangle_area3
    w4 = (pixel_coords[..., 0] * AB[..., 0] - pixel_coords[..., 1] * DA[..., 0]) / twice_triangle_area4

    inside_triangle = (w1 >= 0) & (w2 >= 0) & (w3 >= 0) & (w4 >= 0)
    inside_triangle = inside_triangle.unsqueeze(2)
    
    # Compute pixel color using barycentric coordinates
    pixel_color = pt.stack([red, green, blue], dim=3)
    pixel_color = (pixel_color * inside_triangle).sum(dim=2)
    
    # Update image tensor
    image = image.view(batch_size, 3, -1)
    pixel_indices = (pixel_coords[..., 1] * image_size + pixel_coords[..., 0]).long()
    image.scatter_(2, pixel_indices, pixel_color.view(batch_size, 3, -1))
    image = image.view(batch_size, 3, image_size, image_size)
    
    return image
