"""ATORpt_RFellipsePropertiesClass.py

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

from ATORpt_RFglobalDefs import *
import ATORpt_RFoperations


class EllipsePropertiesClass():	#or EllipsoidProperties
	def __init__(self, centerCoordinates, axesLength, angle, colour):
		self.centerCoordinates = centerCoordinates
		self.axesLength = axesLength
		self.angle = angle
		self.colour = colour	#only used by ATORpt_RFdetectEllipses
		
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
