"""ATORpt_RFoperations.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORpt_RFmainFT.py

# Usage:
See ATORpt_RFmainFT.py

# Description:
ATORpt Operations

"""

import torch as pt
import os
import numpy as np
import cv2
import math
from PIL import Image

from ATORpt_RFglobalDefs import *

class RFresolutionProperties:
	def __init__(self, resolutionIndex, resolutionIndexFirst, numberOfResolutions, imageSizeBase):
		self.resolutionIndex = resolutionIndex
		self.resolutionIndexFirst = resolutionIndexFirst
		self.numberOfResolutions = numberOfResolutions
		self.imageSizeBase = imageSizeBase

		self.resolutionFactor, self.resolutionFactorReverse, self.imageSize = getImageDimensionsR(self)


def modifyTuple(t, index, value):
	lst = list(t)
	lst[index] = value
	tNew = tuple(lst)
	return tNew


def displayImage(outputImage):
	# for debugging purposes only - no way of closing window
	image = cv2.cvtColor(outputImage, cv2.COLOR_BGR2RGB)
	Image.fromarray(image).show()


def saveImage(inputimagefilename, imageObject):
	# outputImageFolder = os.getcwd()	# current folder
	inputImageFolder, inputImageFileName = os.path.split(inputimagefilename)
	outputImageFileName = inputImageFileName
	cv2.imwrite(outputImageFileName, imageObject)


def convertDegreesToRadians(degrees):
	radians = degrees * math.pi / 180
	return radians


def expandDimsN(tensor, numberOfDimensions, axis):
	tensorExpanded = tensor
	for index in range(numberOfDimensions):
		tensorExpanded = pt.unsqueeze(tensorExpanded, dim=axis)
	return tensorExpanded


def rotatePoint2D(origin, point, angle):
	origin1 = (origin[0], origin[1])
	point1 = (point[0], point[1])
	angle1 = angle[0]
	qx1, qy1 = rotatePointNP2D(origin1, point1, angle1)

	pointRotated = (qx1, qy1)
	return pointRotated


def rotatePoint3D(origin, point, angle):
	origin1 = (origin[0], origin[1])
	point1 = (point[0], point[1])
	angle1 = angle[0]
	qx1, qy1 = rotatePointNP2D(origin1, point1, angle1)

	origin2 = (origin1[1], origin[2])
	point2 = (point1[1], point[2])
	angle2 = angle[1]
	qx2, qy2 = rotatePointNP2D(origin2, point2, angle2)

	pointRotated = (qx1, qx2, qy2)
	return pointRotated


def rotatePointNP2D(origin, point, angle):
	theta = convertDegreesToRadians(angle)

	ox, oy = origin
	px, py = point

	qx = ox + math.cos(theta) * (px - ox) - math.sin(theta) * (py - oy)
	qy = oy + math.sin(theta) * (px - ox) + math.cos(theta) * (py - oy)

	return qx, qy


def calculateDistance2D(point1, point2):
	point_a = np.array((point1[0], point1[1]))
	point_b = np.array((point2[0], point2[1]))
	return calculateDistanceNP(point_a, point_b)


def calculateDistance3D(point1, point2):
	point_a = np.array((point1[0], point1[1], point1[2]))
	point_b = np.array((point2[0], point2[1], point2[2]))
	return calculateDistanceNP(point_a, point_b)


def calculateDistanceNP(point1, point2):
	# point1NP = np.asarray(point1)
	# point2NP = np.asarray(point2)
	distance = np.linalg.norm(point1 - point2)
	return distance


def calculateRelativePosition3D(angle, axisLength):
	print("error calculateRelativePosition3D: RFpropertiesParent.numberOfDimensions == 3 not yet coded")
	quit()


def calculateRelativePosition2D(angle, hyp):
	theta = convertDegreesToRadians(angle)
	relativePosition2D = (math.sin(theta) * hyp, math.cos(theta) * hyp)
	return relativePosition2D


def getImageDimensionsR(resolutionProperties):
	# for ATORpt_RFmainCV:
	resolutionIndexReverse = resolutionProperties.numberOfResolutions - resolutionProperties.resolutionIndex + resolutionProperties.resolutionIndexFirst  # CHECKTHIS
	if(ensureMinimumImageSizeGreaterThanRFsize):
		resolutionFactorBase = 4	#minimum imageSize must be >= kernelSize
		resolutionFactor = 2**resolutionIndexReverse / resolutionFactorBase
	else:
		resolutionFactor = 2 ** resolutionIndexReverse

	# for ATORpt:
	resolutionFactorReverse = 2 ** (resolutionProperties.resolutionIndex + 1 - resolutionProperties.resolutionIndexFirst)  # CHECKTHIS
	resolutionFactorInverse = 1.0 / (resolutionFactor)
	# print("resolutionIndex = ", resolutionIndex, ", resolutionFactor = ", resolutionFactor)

	imageSize = (int(resolutionProperties.imageSizeBase[0] / resolutionFactor), int(resolutionProperties.imageSizeBase[1] / resolutionFactor))

	# print("imageSize = ", imageSize)

	return resolutionFactor, resolutionFactorReverse, imageSize


def isTensorEmpty(tensor):
	is_empty = False
	if(tensor.numel() == 0):
		is_empty = True
	return is_empty


def calculateRelativePositionGivenAngleAndLength(angle, length):
	theta = convertDegreesToRadians(angle)
	x = math.cos(theta) * length
	y = math.sin(theta) * length
	# point = [x, y]	# floats unsupported by opencv ellipse draw
	point = [int(x), int(y)]
	return point


def getEquilateralTriangleAxesLength(opp):
	# create equilateral triangle
	# tan(60) = opp/adj
	# adj = tan(60)/opp
	angle = 60
	theta = convertDegreesToRadians(angle)
	adj = math.tan(theta) / opp
	return (opp, adj)

