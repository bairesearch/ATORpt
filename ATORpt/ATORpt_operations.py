"""ATORpt_operations.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt operations

"""

import torch as pt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torchvision.transforms as transforms

from ATORpt_globalDefs import *
if(debugSnapshotRender):
	import ATORpt_PTrenderer
	
def knn_search(points, k):
	points = points.float()
	dist_matrix = pt.cdist(points, points)
	mask = pt.eye(points.shape[0], dtype=pt.bool)	#ensure that sample keypoint is not added to nearest keypoints
	dist_matrix[mask] = float('inf')
	pointsKnearest = pt.topk(dist_matrix, k=k, dim=1, largest=False)
	pointsKnearestValues = pointsKnearest.values
	k_nearest_points = points[pointsKnearest.indices]
	#print("points = ", points)
	#print("k_nearest_points = ", k_nearest_points)
	return k_nearest_points

def printImage(image, isMonochrome=False):
	if(not isMonochrome):
		image = image.permute(1, 2, 0)	#place image into H,W,C format
	image = image.cpu().numpy().squeeze()
	if(isMonochrome):
		plt.imshow(image, cmap='gray')
	else:
		plt.imshow(image)
	plt.show()

def printFeatureMap(posEmbeddings, featureMapN):
	if(debugGeometricHashingParallel):
		print("printFeatureMap")
		printImageCoordinates(posEmbeddings[:, xAxisFeatureMap], posEmbeddings[:, yAxisFeatureMap], featureMapN)

def printPixelMap(posEmbeddings, tokens):
	if(debugGeometricHashingParallel):
		print("printPixelMap")
		firstIndexInBatch = 0
		printImageCoordinates(posEmbeddings[:, xAxisFeatureMap], posEmbeddings[:, yAxisFeatureMap], tokens[firstIndexInBatch])

def printImageCoordinates(x, y, values, imageSize=700):

	colorGraph = False
	if(len(values.shape) > 1):
		colorGraph = True
		values = values.permute(1, 0)
		
	#print("x = ", x)
	#print("y = ", y)
	#print("values = ", values)

	plotX = x.cpu().detach().numpy()
	plotY = y.cpu().detach().numpy()
	plotZ = values.cpu().detach().numpy()

	if(debugGeometricHashingParallelLargeMarker):
		markerSize = 1
	else:
		markerSize = 0.01
	plt.subplot(121)

	if(colorGraph):
		plt.scatter(x=plotX, y=plotY, c=plotZ, s=markerSize)
	else:
		plotZ = 1.0-plotZ	#invert such that MNIST number pixels are displayed as black (on white background)
		plt.scatter(x=plotX, y=plotY, c=plotZ, s=markerSize, vmin=0, vmax=1, cmap=cm.gray)	#assume input is normalised (0->1.0) #unnormalised (0 -> 255)

	plt.xlim(-imageSize, imageSize)
	plt.ylim(-imageSize, imageSize)
	
	plt.gca().set_aspect('equal', adjustable='box')

	plt.show()	
	

def printCoordinates(keypointCoordinates, meshCoordinates, meshValues, meshFaces):
	printKeypoints(keypointCoordinates)
	printPixelCoordinates(meshCoordinates, meshValues, meshFaces)
	
def printKeypoints(keypointCoordinates):
	if(debugGeometricHashingParallel):
		print("printKeypoints")
		keypointCoordinatesCombined = pt.reshape(keypointCoordinates, (keypointCoordinates.shape[0]*keypointCoordinates.shape[1], keypointCoordinates.shape[2]))	#combine keyPointA/keyPointB/keyPointC
		keypointValuesCombined = pt.ones(keypointCoordinatesCombined[:, xAxisGeometricHashing].shape)
		#print("keypointCoordinatesCombined.shape = ", keypointCoordinatesCombined.shape)
		#print("keypointValuesCombined.shape = ", keypointValuesCombined.shape)
		printImageCoordinates(keypointCoordinatesCombined[:, xAxisGeometricHashing], keypointCoordinatesCombined[:, yAxisGeometricHashing], keypointValuesCombined)

def printPixelCoordinates(meshCoordinates, meshValues, meshFaces):
	'''
	if(debugGeometricHashingParallel):			
		print("printPixelCoordinates:")
		#print("meshCoordinates.shape = ", meshCoordinates.shape)
		#print("meshValues.shape = ", meshValues.shape)
		printImageCoordinates(meshCoordinates[:, :, xAxisGeometricHashing], meshCoordinates[:, :, yAxisGeometricHashing], meshValues)
	'''
	if(debugSnapshotRender):
		print("printPixelCoordinates:")
		renderViewportSizeDebug = renderViewportSize
		renderImageSizeDebug = renderViewportSize
		transformedPatches = ATORpt_PTrenderer.resamplePixelCoordinates(meshCoordinates, meshValues, meshFaces, renderViewportSizeDebug, renderImageSizeDebug, centreSnapshots=True)
		
def printCoordinatesIndex(keypointCoordinates, meshCoordinates, meshValues, meshFaces, index, step=None):
	printKeypointsIndex(keypointCoordinates, index, step)
	printPixelCoordinatesIndex(meshCoordinates, meshValues, meshFaces, index, step)
	
def printKeypointsIndex(keypointCoordinates, index, step=None):
	if(debugGeometricHashingParallel):
		if(step < 1):	#before final scale transform
			debugPlotImageSize = 700
		elif(step < 4):	#before final scale transform
			debugPlotImageSize = 50
		else:
			debugPlotImageSize = renderViewportSize[0]*2
		print("printKeypointsIndex: step=" + str(step))
		print("keypointCoordinates[index] = ", keypointCoordinates[index])
		keypointCoordinatesCombined = keypointCoordinates[index, :, :]
		keypointValuesCombined = pt.ones(keypointCoordinatesCombined[:, xAxisGeometricHashing].shape)
		printImageCoordinates(keypointCoordinatesCombined[:, xAxisGeometricHashing], keypointCoordinatesCombined[:, yAxisGeometricHashing], keypointValuesCombined[:], debugPlotImageSize)

def printPixelCoordinatesIndex(meshCoordinates, meshValues, meshFaces, index, step=None):
	'''
	if(debugGeometricHashingParallel):			
		print("printPixelCoordinatesIndex: step=" + str(step))
		#print("meshCoordinates.shape = ", meshCoordinates.shape)
		#print("meshValues.shape = ", meshValues.shape)
		#meshValues = meshValues*0.5
		printImageCoordinates(meshCoordinates[index, :, xAxisGeometricHashing], meshCoordinates[index, :, yAxisGeometricHashing], meshValues[index])	
	'''
	if(debugSnapshotRender):
		if(step < 1):	#before final scale transform
			renderViewportSizeDebug = (700, 700)
		elif(step < 4):	#before final scale transform
			renderViewportSizeDebug = (50, 50)
		else:
			renderViewportSizeDebug = renderViewportSize
		renderImageSizeDebug = 256
		print("printPixelCoordinatesIndex: step=" + str(step))
		transformedPatches = ATORpt_PTrenderer.resamplePixelCoordinates(meshCoordinates, meshValues, meshFaces, renderViewportSizeDebug, renderImageSizeDebug, centreSnapshots=True, index=index)




def pil_to_tensor(image):
	transformToTensor = transforms.ToTensor()
	#transformToTensor = transforms.Compose([transforms.ToTensor(),])
	return transformToTensor(image)
	
