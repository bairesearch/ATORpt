"""ATORpt_RFellipseProperties.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt Ellipse (or Ellipsoid) Properties

"""

import numpy as np
import cv2
import copy

import ATORpt_RFoperations

ellipseAngleResolution = 10	#degrees
minimumEllipseFitErrorRequirement = 1500.0	#calibrate



class EllipsePropertiesClass():	#or EllipsoidProperties
	def __init__(self, centerCoordinates, axesLength, angle, colour):
		self.centerCoordinates = centerCoordinates
		self.axesLength = axesLength
		self.angle = angle
		self.colour = colour	#only used by ATORpt_RFdetectEllipses
	
def drawEllipse(outputImage, ellipseProperties, relativeCoordiantes):
	#https://docs.opencv.org/4.5.3/d6/d6e/group__imgproc__draw.html#ga28b2267d35786f5f890ca167236cbc69
	#print("ellipseProperties.centerCoordinates = ", ellipseProperties.centerCoordinates)
	#print("ellipseProperties.axesLength = ", ellipseProperties.axesLength)
	#print("ellipseProperties.angle = ", ellipseProperties.angle)
	#print("ellipseProperties.colour = ", ellipseProperties.colour)
	
	centerCoordinates = getAbsoluteImageCenterCoordinates(outputImage, ellipseProperties, relativeCoordiantes)	
	#print("centerCoordinates = ", centerCoordinates)
	
	cv2.ellipse(outputImage, centerCoordinates, ellipseProperties.axesLength, ellipseProperties.angle, 0, 360, ellipseProperties.colour, -1)
	
	#print("outputImage = ", outputImage)
		
def drawCircle(outputImage, ellipseProperties, relativeCoordiantes):	
	#https://docs.opencv.org/4.5.3/d6/d6e/group__imgproc__draw.html#gaf10604b069374903dbd0f0488cb43670
	
	centerCoordinates = getAbsoluteImageCenterCoordinates(outputImage, ellipseProperties, relativeCoordiantes)
	
	cv2.circle(outputImage, centerCoordinates, ellipseProperties.axesLength[0], ellipseProperties.colour, -1)
	
	#print("outputImage = ", outputImage)

def drawRectangle(outputImage, ellipseProperties, relativeCoordiantes):	
	#https://docs.opencv.org/4.5.3/d6/d6e/group__imgproc__draw.html#ga07d2f74cadcf8e305e810ce8eed13bc9
	
	centerCoordinates = getAbsoluteImageCenterCoordinates(outputImage, ellipseProperties, relativeCoordiantes)
	
	#print("ellipseProperties.axesLength[0] = ", ellipseProperties.axesLength[0])
	
	point1 = (centerCoordinates[0]-ellipseProperties.axesLength[0], centerCoordinates[1]-ellipseProperties.axesLength[1])
	point2 = (centerCoordinates[0]+ellipseProperties.axesLength[0], centerCoordinates[1]+ellipseProperties.axesLength[1])
	cv2.rectangle(outputImage, point1, point2, ellipseProperties.colour, -1)
	
	#print("outputImage = ", outputImage)
	
	
def drawPoint(outputImage, ellipseProperties, relativeCoordiantes):		
	centerCoordinates = getAbsoluteImageCenterCoordinates(outputImage, ellipseProperties, relativeCoordiantes)
	
	#print("ellipseProperties.centerCoordinates = ", ellipseProperties.centerCoordinates)
	#print("centerCoordinates = ", centerCoordinates)
	
	x = centerCoordinates[0]
	y = centerCoordinates[1]
	outputImage[y, x, 0] = ellipseProperties.colour[0]
	
	#print("outputImage = ", outputImage)
		
	
def getAbsoluteImageCenterCoordinates(outputImage, ellipseProperties, relativeCoordiantes):
	if(relativeCoordiantes):
		imageSize = outputImage.shape
		#print("imageSize = ", imageSize)
		centerCoordinates = (ellipseProperties.centerCoordinates[0]+int(imageSize[0]/2), ellipseProperties.centerCoordinates[1]+int(imageSize[1]/2))
	else:
		centerCoordinates = ellipseProperties.centerCoordinates
	return centerCoordinates
		
def normaliseGlobalEllipseProperties(ellipseProperties, resolutionFactor):
	resolutionFactor = ellipseProperties.resolutionFactor
	ellipsePropertiesNormalised = copy.deepcopy(ellipseProperties) 
	ellipsePropertiesNormalised.centerCoordinates = (ellipsePropertiesNormalised.centerCoordinates[0]*resolutionFactor, ellipsePropertiesNormalised.centerCoordinates[1]*resolutionFactor)
	ellipsePropertiesNormalised.axesLength = (ellipsePropertiesNormalised.axesLength[0]*resolutionFactor, ellipsePropertiesNormalised.axesLength[1]*resolutionFactor)
	return ellipsePropertiesNormalised
	 
def calculateEllipseFitError(inputImage, inputImageMod):
	meanSquaredError = (np.sqrt((pow(np.subtract(inputImage, inputImageMod, dtype=np.int32), 2))).sum())	#currently use mean squared error
	ellipseFitError = meanSquaredError
	return ellipseFitError

def testEllipseApproximation(inputImageR, ellipseProperties):
	inputImageRmod = copy.deepcopy(inputImageR)
	cv2.ellipse(inputImageRmod, ellipseProperties.centerCoordinates, ellipseProperties.axesLength, ellipseProperties.angle, 0, 360, ellipseProperties.colour, -1)
	ellipseFitError = calculateEllipseFitError(inputImageR, inputImageRmod)
	return inputImageRmod, ellipseFitError

def centroidOverlapsEllipseWrapper(ellipseFitError, ellipseProperties, ellipsePropertiesOptimumLast):
	result = True
	if(ellipseFitError < minimumEllipseFitErrorRequirement):
		if ellipsePropertiesOptimumLast is None:
			result = False
		else:
			result = centroidOverlapsEllipse(ellipseProperties, ellipsePropertiesOptimumLast)
	return result					

def centroidOverlapsEllipse(ellipseProperties, ellipsePropertiesOptimumLast):
	result = True
	#minimumDistance = max(ellipseProperties.axesLength[0], ellipseProperties.axesLength[1], ellipsePropertiesOptimumLast.axesLength[0], ellipsePropertiesOptimumLast.axesLength[1])	#CHECKTHIS
	minimumDistance = max((ellipseProperties.axesLength[0] + ellipseProperties.axesLength[1])/2, (ellipsePropertiesOptimumLast.axesLength[0] + ellipsePropertiesOptimumLast.axesLength[1])/2)	#CHECKTHIS
	if((ellipsePropertiesOptimumLast.centerCoordinates[0] - ellipseProperties.centerCoordinates[0])**2 + (ellipsePropertiesOptimumLast.centerCoordinates[1] - ellipseProperties.centerCoordinates[1])**2) > minimumDistance**2:
		result = False
	return result		
	
def printEllipseProperties(ellipseProperties):
	print("printEllipseProperties: centerCoordinates = ", ellipseProperties.centerCoordinates, ", axesLength = ", ellipseProperties.axesLength, ", angle = ", ellipseProperties.angle, ", colour = ", ellipseProperties.colour)	
