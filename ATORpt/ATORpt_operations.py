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

def printImage(image):
	image = image.cpu().numpy().squeeze()
	plt.imshow(image, cmap='gray')
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

def printImageCoordinates(x, y, values):

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

	if(debugGeometricHashingParallel2):
		markerSize = 1
	else:
		markerSize = 0.01
	plt.subplot(121)

	if(colorGraph):
		plt.scatter(x=plotX, y=plotY, c=plotZ, s=markerSize)
	else:
		plotZ = 1.0-plotZ	#invert such that MNIST number pixels are displayed as black (on white background)
		plt.scatter(x=plotX, y=plotY, c=plotZ, s=markerSize, vmin=0, vmax=1, cmap=cm.gray)	#assume input is normalised (0->1.0) #unnormalised (0 -> 255)

	if(debugGeometricHashingParallel2):
		plt.xlim(-1, 1)
		plt.ylim(-1, 1)
	else:
		plt.xlim(0, 700)
		plt.ylim(0, 700)
	plt.gca().set_aspect('equal', adjustable='box')

	plt.show()	
	

def printKeypoints(keypointCoordinates):
	if(debugGeometricHashingParallel):
		print("printKeypoints")
		keypointCoordinatesCombined = pt.reshape(keypointCoordinates, (keypointCoordinates.shape[0]*keypointCoordinates.shape[1], keypointCoordinates.shape[2]))	#combine keyPointA/keyPointB/keyPointC
		keypointValuesCombined = pt.ones(keypointCoordinatesCombined[:, xAxisGeometricHashing].shape)
		#print("keypointCoordinatesCombined.shape = ", keypointCoordinatesCombined.shape)
		#print("keypointValuesCombined.shape = ", keypointValuesCombined.shape)
		printImageCoordinates(keypointCoordinatesCombined[:, xAxisGeometricHashing], keypointCoordinatesCombined[:, yAxisGeometricHashing], keypointValuesCombined)

def printKeypointsIndex(keypointCoordinates, index):
	if(debugGeometricHashingParallel):
		print("printKeypointsIndex")
		#print("keypointCoordinates = ", keypointCoordinates)
		keypointCoordinatesCombined = keypointCoordinates[index, :, :]
		keypointValuesCombined = pt.ones(keypointCoordinatesCombined[:, xAxisGeometricHashing].shape)
		#print("keypointCoordinatesCombined = ", keypointCoordinatesCombined)
		#print("keypointValuesCombined = ", keypointValuesCombined)
		printImageCoordinates(keypointCoordinatesCombined[:, xAxisGeometricHashing], keypointCoordinatesCombined[:, yAxisGeometricHashing], keypointValuesCombined[:])

def printPixelCoordinates(pixelCoordinates, pixelValues):
	if(debugGeometricHashingParallel):			
		print("printPixelCoordinates")
		#print("pixelCoordinates.shape = ", pixelCoordinates.shape)
		#print("pixelValues.shape = ", pixelValues.shape)
		printImageCoordinates(pixelCoordinates[:, :, xAxisGeometricHashing], pixelCoordinates[:, :, yAxisGeometricHashing], pixelValues)

def printPixelCoordinatesIndex(pixelCoordinates, pixelValues, index, text=None):
	if(debugGeometricHashingParallel):			
		print("printPixelCoordinatesIndex: " + text)
		#print("pixelCoordinates.shape = ", pixelCoordinates.shape)
		#print("pixelValues.shape = ", pixelValues.shape)
		#pixelValues = pixelValues*0.5
		printImageCoordinates(pixelCoordinates[index, :, xAxisGeometricHashing], pixelCoordinates[index, :, yAxisGeometricHashing], pixelValues[index])	


def pil_to_tensor(image):
	transformToTensor = transforms.ToTensor()
	#transformToTensor = transforms.Compose([transforms.ToTensor(),])
	return transformToTensor(image)
	
