"""ATORpt_RFgenerateDraw.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORpt_RFmain.py

# Usage:
See ATORpt_RFmain.py

# Description:
ATORpt Generate Draw

"""

import torch as pt
import numpy as np
import cv2
import copy

from ATORpt_RFglobalDefs import *

def drawRF(RFfilterTF, RFpropertiesInside, RFpropertiesOutside, drawFeatureType, drawFeatureOverlay):
	blankArray = np.full((RFpropertiesInside.imageSize[1], RFpropertiesInside.imageSize[0], 1), 0, np.uint8)
	ellipseFilterImageInside = copy.deepcopy(blankArray)
	ellipseFilterImageOutside = copy.deepcopy(blankArray)
	RFpropertiesInsideWhite = copy.deepcopy(RFpropertiesInside)
	RFpropertiesInsideWhite.colour = (255, 255, 255)
	RFpropertiesOutsideWhite = copy.deepcopy(RFpropertiesOutside)
	RFpropertiesOutsideWhite.colour = (255, 255, 255)
	RFpropertiesInsideBlack = copy.deepcopy(RFpropertiesInside)
	RFpropertiesInsideBlack.colour = (000, 000, 000)
	if drawFeatureType == RFfeatureTypeEllipse:
		drawEllipse(ellipseFilterImageInside, RFpropertiesInsideWhite, True)
		drawEllipse(ellipseFilterImageOutside, RFpropertiesOutsideWhite, True)
		drawEllipse(ellipseFilterImageOutside, RFpropertiesInsideBlack, True)
	elif drawFeatureType == RFfeatureTypeCircle:
		drawCircle(ellipseFilterImageInside, RFpropertiesInsideWhite, True)
		drawCircle(ellipseFilterImageOutside, RFpropertiesOutsideWhite, True)
		drawCircle(ellipseFilterImageOutside, RFpropertiesInsideBlack, True)
	elif drawFeatureType == RFfeatureTypePoint:
		drawPoint(ellipseFilterImageInside, RFpropertiesInsideWhite, True)
		drawCircle(ellipseFilterImageOutside, RFpropertiesOutsideWhite, True)
		drawPoint(ellipseFilterImageOutside, RFpropertiesInsideBlack, True)
	elif drawFeatureType == RFfeatureTypeCorner:
		drawPoint(ellipseFilterImageInside, RFpropertiesInsideWhite, True)
		drawRectangle(ellipseFilterImageOutside, RFpropertiesOutsideWhite, True)
		drawPoint(ellipseFilterImageOutside, RFpropertiesInsideBlack, True)
	insideImageTF = pt.tensor(ellipseFilterImageInside, dtype=pt.float32)
	insideImageTF = pt.gt(insideImageTF, 0.0)
	insideImageTF = insideImageTF.type(pt.float32)
	outsideImageTF = pt.tensor(ellipseFilterImageOutside, dtype=pt.float32)
	outsideImageTF = pt.gt(outsideImageTF, 0.0)
	outsideImageTF = outsideImageTF.type(pt.float32)
	multiples = tuple([1, 1, 3])
	insideImageTF = insideImageTF.repeat(multiples)
	RFcolourInside = pt.tensor([RFpropertiesInside.colour[0], RFpropertiesInside.colour[1], RFpropertiesInside.colour[2]], dtype=pt.float32)
	RFcolourInside = RFcolourInside.unsqueeze(dim=0)
	insideImageTF = insideImageTF * RFcolourInside
	multiples = tuple([1, 1, 3])
	outsideImageTF = outsideImageTF.repeat(multiples)
	RFcolourOutside = pt.tensor([RFpropertiesOutside.colour[0], RFpropertiesOutside.colour[1], RFpropertiesOutside.colour[2]], dtype=pt.float32)
	RFcolourOutside = RFcolourOutside.unsqueeze(dim=0)
	outsideImageTF = outsideImageTF * RFcolourOutside
	if drawFeatureOverlay:
		RFfilterTFon = pt.ne(RFfilterTF, 0.0)
		insideImageTFon = pt.ne(insideImageTF, 0.0)
		outsideImageTFon = pt.ne(outsideImageTF, 0.0)
		insideImageTFonOverlap = pt.logical_and(RFfilterTFon, insideImageTFon)
		outsideImageTFonOverlap = pt.logical_and(RFfilterTFon, outsideImageTFon)
		insideImageTFonMask = pt.logical_not(insideImageTFonOverlap).type(pt.float32)
		outsideImageTFonMask = pt.logical_not(outsideImageTFonOverlap).type(pt.float32)
		insideImageTF = insideImageTF * insideImageTFonMask
		outsideImageTF = outsideImageTF * outsideImageTFonMask
		RFfilterTF = RFfilterTF + insideImageTF + outsideImageTF
	else:
		RFfilterTF = RFfilterTF + insideImageTF + outsideImageTF
	return RFfilterTF
	
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
	
	#print("centerCoordinates[0] = ", centerCoordinates[0])
	#print("centerCoordinates[1] = ", centerCoordinates[1])
	#print("ellipseProperties.axesLength[0] = ", ellipseProperties.axesLength[0])
	
	point1 = (centerCoordinates[0]-ellipseProperties.axesLength[0], centerCoordinates[1]-ellipseProperties.axesLength[1])
	point2 = (centerCoordinates[0]+ellipseProperties.axesLength[0], centerCoordinates[1]+ellipseProperties.axesLength[1])
	cv2.rectangle(outputImage, point1, point2, ellipseProperties.colour, -1)
	
	#print("outputImage = ", outputImage)
	
	
def drawPoint(outputImage, ellipseProperties, relativeCoordiantes):		
	centerCoordinates = getAbsoluteImageCenterCoordinates(outputImage, ellipseProperties, relativeCoordiantes)
	
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
