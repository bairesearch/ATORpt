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

def printImageCoordinates(x, y, values, imageSize=700, permuteColorValues=True, title=""):

	figsize=(10, 10) # Width, Height in inches
	
	colorGraph = False
	if(len(values.shape) > 1):
		colorGraph = True
		if(permuteColorValues):
			values = values.permute(1, 0)
		
	#print("x = ", x)
	#print("y = ", y)
	#print("values = ", values)

	plotX = x.cpu().detach().numpy()
	plotY = y.cpu().detach().numpy()
	plotC = values.cpu().detach().numpy()

	if(debugGeometricHashingParallelLargeMarker):
		markerSize = 1
	else:
		markerSize = 0.01
	
	plt.figure(figsize=figsize)
	if(colorGraph):
		plt.scatter(x=plotX, y=plotY, c=plotC, s=markerSize)
	else:
		plotZ = 1.0-plotZ	#invert such that MNIST number pixels are displayed as black (on white background)
		plt.scatter(x=plotX, y=plotY, c=plotC, s=markerSize, vmin=0, vmax=1, cmap=cm.gray)	#assume input is normalised (0->1.0) #unnormalised (0 -> 255)
	plt.xlim(-imageSize, imageSize)
	plt.ylim(-imageSize, imageSize)
	
	fig = plt.gcf()
	fig.canvas.manager.set_window_title(title)
	fig.set_size_inches(figsize)  
	plt.gca().set_aspect('equal', adjustable='box')
	plt.show()	

def printCoordinates(use3DOD, keypointCoordinates, meshCoordinates, meshValues, meshFaces, step=None, centreSnapshots=True):
	if(debugGeometricHashingParallelFinal):
		printKeypoints(use3DOD, keypointCoordinates, step)
	if(debugSnapshotRenderFinal):
		printPixelCoordinates(use3DOD, meshCoordinates, meshValues, meshFaces, step=step, centreSnapshots=centreSnapshots)
	
def printKeypoints(use3DOD, keypointCoordinates, step=None):
	for index in range(keypointCoordinates.shape[0]):
		print("printKeypoints: index = ", index)
		printKeypointsIndex(use3DOD, keypointCoordinates, index, step=step)

def printPixelCoordinates(use3DOD, meshCoordinates, meshValues, meshFaces, step=None, centreSnapshots=True):
	if(step is None):
		print("printPixelCoordinates:")
		renderViewportSizeDebug = renderViewportSize
		renderImageSizeDebug = renderImageSize
	else:
		print("printPixelCoordinates: step=" + str(step))
		if(step < 1):	#before final scale transform
			renderViewportSizeDebug = (700, 700)	#max image size
			renderImageSizeDebug = 1000	#256
		elif(step < 5):	#before final scale transform
			renderViewportSizeDebug = ATORpatchSizeIntermediary
			renderImageSizeDebug = 1000	#256
		else:
			renderViewportSizeDebug = renderViewportSize	#*2 for debug checking only
			renderImageSizeDebug = renderImageSize	#*2 for debug checking only
	transformedPatches = ATORpt_PTrenderer.resamplePixelCoordinates(use3DOD, meshCoordinates, meshValues, meshFaces, renderViewportSizeDebug, renderImageSizeDebug, centreSnapshots=centreSnapshots)

def printCoordinatesIndex(use3DOD, keypointCoordinates, meshCoordinates, meshValues, meshFaces, index, step=None):
	if(debugGeometricHashingParallel):
		printKeypointsIndex(use3DOD, keypointCoordinates, index, step)
	if(debugSnapshotRender):
		printPixelCoordinatesIndex(use3DOD, meshCoordinates, meshValues, meshFaces, index, step)
	
def printKeypointsIndex(use3DOD, keypointCoordinates, index, step=None):
	if(step < 1):	#before final scale transform
		debugPlotImageSize = 700	#max image size
	elif(step < 4):	#before final scale transform
		debugPlotImageSize = ATORpatchSizeIntermediary[xAxisGeometricHashing]
	else:
		debugPlotImageSize = renderViewportSize[xAxisGeometricHashing]*2	#*2 for debug checking only
	print("printKeypointsIndex: step=" + str(step))
	print("keypointCoordinates[index] = ", keypointCoordinates[index])
	keypointCoordinatesCombined = keypointCoordinates[index, :, :]
	#keypointValues = pt.ones(keypointCoordinatesCombined[:, xAxisGeometricHashing].shape)
	keypointValues = pt.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) #kp0/A:Red, kp1/B:Green, kp2/C:Blue
	title = "poly index: " + str(index)
	printImageCoordinates(keypointCoordinatesCombined[:, xAxisGeometricHashing], keypointCoordinatesCombined[:, yAxisGeometricHashing], keypointValues, imageSize=debugPlotImageSize, permuteColorValues=False, title=title)

def printPixelCoordinatesIndex(use3DOD, meshCoordinates, meshValues, meshFaces, index, step=None, centreSnapshots=True):
	if(step < 1):	#before final scale transform
		renderViewportSizeDebug = (700, 700)	#max image size
		renderImageSizeDebug = 1000	#256
	elif(step < 5):	#before final scale transform
		renderViewportSizeDebug = ATORpatchSizeIntermediary
		renderImageSizeDebug = 1000	#256
	else:
		renderViewportSizeDebug = renderViewportSize	#*2 for debug checking only
		renderImageSizeDebug = renderImageSize	#*2 for debug checking only
	#print("printPixelCoordinatesIndex: step=" + str(step))
	transformedPatches = ATORpt_PTrenderer.resamplePixelCoordinates(use3DOD, meshCoordinates, meshValues, meshFaces, renderViewportSizeDebug, renderImageSizeDebug, centreSnapshots=centreSnapshots, index=index)

def pil_to_tensor(image):
	transformToTensor = transforms.ToTensor()
	#transformToTensor = transforms.Compose([transforms.ToTensor(),])
	return transformToTensor(image)
	
