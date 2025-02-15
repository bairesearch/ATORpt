"""ATORpt_RFmainCV.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORpt_RFmainFT.py

# Usage:
source activate pytorch3d
python ATORpt_RFmainCV.py images/leaf1.png

# Description:
Perform ATOR receptive field (RF) ellipse detection using open-cv (CV) library (non-hardware accelerated) rather than RF filters.

See [Vinmorel](https://github.com/vinmorel/Genetic-Algorithm-Image) for genetic algorithm implementation.

"""


import cv2
import copy
#import click
import numpy as np
from collections import OrderedDict

from ATORpt_RFglobalDefs import *
import ATORpt_RFoperations
import ATORpt_RFellipsePropertiesClass
import ATORpt_RFgenerateDraw

def detectEllipsesGaussianBlur(inputimagefilename):
	
	inputImage = cv2.imread(inputimagefilename)
	inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
	
	inputImageHeight, inputImageWidth, inputImageChannels = inputImage.shape
	inputImageSize = (inputImageWidth, inputImageHeight)
	print("inputImageHeight = ", inputImageHeight, "inputImageWidth = ", inputImageWidth, ", inputImageChannels = ", inputImageChannels)
	blankArray = np.full((inputImageHeight, inputImageWidth, 3), 255, np.uint8)
	outputImage = blankArray
		
	ellipsePropertiesOptimumNormalisedAllRes = []
	
	testEllipseIndex = 0
	
	for resolutionIndex in range(ellipseResolutionIndexFirst, ellipseNumberOfResolutions):
		
		resolutionProperties = ATORpt_RFoperations.RFresolutionProperties(resolutionIndex, ellipseResolutionIndexFirst, ellipseNumberOfResolutions, inputImageSize)
		(resolutionFactor, resolutionFactorReverse, imageSize) = (resolutionProperties.resolutionFactor, resolutionProperties.resolutionFactorReverse, resolutionProperties.imageSize)
	
		#gaussianBlurKernelSize = (resolutionIndexReverse*2) - 1		
		gaussianBlurKernelSize = (resolutionFactor*2) - 1	#ensure kernel size is odd
		print("gaussianBlurKernelSize = ", gaussianBlurKernelSize)
		inputImageR = gaussianBlur(inputImage, gaussianBlurKernelSize)
		
		#inputImageR = cv2.resize(inputImage, None, fx=resolutionFactorInverse, fy=resolutionFactorInverse)
		
		imageHeight, imageWidth, imageChannels = inputImageR.shape
		print("resolutionFactor = ", resolutionFactor, ", imageHeight = ", imageHeight, "imageWidth = ", imageWidth, ", imageChannels = ", imageChannels)
		
		thresh = cv2.cvtColor(inputImageR, cv2.COLOR_RGB2GRAY)
		#ATORpt_RFoperations.displayImage(inputImageR)
		thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 101, 0)
		if(ATORpt_RFoperations.opencvVersion==3):
			NULL, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		elif(ATORpt_RFoperations.opencvVersion==4):
			contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)	#or RETR_TREE
		
		#ret, thresh = cv2.threshold(thresh, 127, 255, cv2.THRESH_BINARY_INV)	#or cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU #binarize
		#contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)	#or RETR_TREE	
		
		minimumArea = 500
		
		inputImageRdev = copy.deepcopy(inputImageR)
		cv2.drawContours(inputImageRdev, contours, -1, (0, 255, 0), 3)
		
		for cnt in contours:
			area = cv2.contourArea(cnt)
			if(area > minimumArea):
				print("cnt")
				averageColour = calculateAverageColourOfContour(inputImageR, cnt)
				ellipse = cv2.fitEllipse(cnt)
				cv2.ellipse(inputImageRdev, ellipse, averageColour, 3)

		ATORpt_RFoperations.displayImage(inputImageRdev)	#debug
																
def detectEllipsesTrialResize(inputimagefilename):
	
	inputImage = cv2.imread(inputimagefilename)
	inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
		
	inputImageHeight, inputImageWidth, inputImageChannels = inputImage.shape
	inputImageSize = (inputImageWidth, inputImageHeight)
	print("inputImageHeight = ", inputImageHeight, "inputImageWidth = ", inputImageWidth, ", inputImageChannels = ", inputImageChannels)
	blankArray = np.full((inputImageHeight, inputImageWidth, 3), 255, np.uint8)
	outputImage = blankArray
	
	ellipsePropertiesOptimumNormalisedAllRes = []
	
	testEllipseIndex = 0
	
	for resolutionIndex in range(ellipseResolutionIndexFirst, ellipseNumberOfResolutions):
	
		resolutionProperties = ATORpt_RFoperations.RFresolutionProperties(resolutionIndex, ellipseResolutionIndexFirst, ellipseNumberOfResolutions, inputImageSize)
		(resolutionFactor, resolutionFactorReverse, imageSize) = (resolutionProperties.resolutionFactor, resolutionProperties.resolutionFactorReverse, resolutionProperties.imageSize)
		
		resolutionFactorInverse = 1.0/(resolutionFactor)
		#print("resolutionIndex = ", resolutionIndex, ", resolutionFactor = ", resolutionFactor)
		inputImageR = cv2.resize(inputImage, None, fx=resolutionFactorInverse, fy=resolutionFactorInverse)
		imageHeight, imageWidth, imageChannels = inputImageR.shape

		#ATORpt_RFoperations.displayImage(inputImageR)	#debug

		print("resolutionFactor = ", resolutionFactor, ", imageHeight = ", imageHeight, "imageWidth = ", imageWidth, ", imageChannels = ", imageChannels)
		
		#match multiple ellipses for each resolution
		ellipsePropertiesOrderedDict = OrderedDict()
		
		#reduce max size of ellipse at each res
		#axesLengthMax1 = imageWidth
		#axesLengthMax2 = imageHeight
		axesLengthMax1 = imageWidth//resolutionFactorReverse * 4	#CHECKTHIS
		axesLengthMax2 = imageHeight//resolutionFactorReverse * 4	#CHECKTHIS
		print("axesLengthMax1 = ", axesLengthMax1, ", axesLengthMax2 = ", axesLengthMax2)
		
		for centerCoordinates1 in range(0, imageWidth, ellipseCenterCoordinatesResolution):
			for centerCoordinates2 in range(0, imageHeight, ellipseCenterCoordinatesResolution):
				for axesLength1 in range(ellipseMinimumEllipseAxisLength, axesLengthMax1, ellipseAxesLengthResolution):
					for axesLength2 in range(ellipseMinimumEllipseAxisLength, axesLengthMax2, ellipseAxesLengthResolution):
						for angle in range(0, 360, ellipseAngleResolution):	#degrees
							for colour1 in range(0, 256, ellipseColourResolution):
								for colour2 in range(0, 256, ellipseColourResolution):
									for colour3 in range(0, 256, ellipseColourResolution):

										imageSize = (imageWidth, imageHeight)
										centerCoordinates = (centerCoordinates1, centerCoordinates2)
										axesLength = (axesLength1, axesLength2)
										colour = (colour1, colour2, colour3)
										
										ellipseProperties = ATORpt_RFellipsePropertiesClass.EllipsePropertiesClass(centerCoordinates, axesLength, angle, colour)
										inputImageRmod, ellipseFitError = ATORpt_RFellipsePropertiesClass.testEllipseApproximation(inputImageR, ellipseProperties)
	
										ellipsePropertiesOrderedDict[ellipseFitError] = ellipseProperties
										testEllipseIndex = testEllipseIndex + 1
										
										#ATORpt_RFellipsePropertiesClass.printEllipseProperties(ellipseProperties)

																									
		ellipsePropertiesOptimumNormalisedR = []
		for ellipseFitError, ellipseProperties in ellipsePropertiesOrderedDict.items():
			
			ellipsePropertiesNormalised = ATORpt_RFellipsePropertiesClass.normaliseGlobalEllipseProperties(ellipseProperties, resolutionFactor)
			
			ellipseOverlapsesWithPreviousOptimumEllipse = False
			for ellipseProperties2 in ellipsePropertiesOptimumNormalisedR:
				if(ATORpt_RFellipsePropertiesClass.centroidOverlapsEllipseWrapper(ellipseFitError, ellipsePropertiesNormalised, ellipseProperties2)):
					ellipseOverlapsesWithPreviousOptimumEllipse = True
						
			if(not ellipseOverlapsesWithPreviousOptimumEllipse):
				ellipsePropertiesNormalisedOptimumLast = ellipsePropertiesNormalised
				ellipsePropertiesOptimumNormalisedAllRes.append(ellipsePropertiesNormalisedOptimumLast)
				ellipsePropertiesOptimumNormalisedR.append(ellipsePropertiesNormalisedOptimumLast)
				#inputImageRmod, ellipseFitError = ATORpt_RFellipsePropertiesClass.testEllipseApproximation(inputImageR, ellipseProperties)
				outputImage = ATORpt_RFgenerateDraw.drawEllipse(outputImage, ellipsePropertiesNormalisedOptimumLast, False)
				ATORpt_RFoperations.displayImage(outputImage)
				ATORpt_RFoperations.saveImage(inputimagefilename, outputImage)

		#quit()

def gaussianBlur(inputImage, gaussianBlurKernelSize):
	gaussianBlurKernelSize = int(gaussianBlurKernelSize)  # Ensure integer
	gaussianBlurKernelSize = gaussianBlurKernelSize if gaussianBlurKernelSize % 2 == 1 else gaussianBlurKernelSize + 1  # Ensure odd
	
	result = cv2.GaussianBlur(src=inputImage, ksize=(gaussianBlurKernelSize,gaussianBlurKernelSize), sigmaX=20.0, borderType=cv2.BORDER_DEFAULT)
	return result
	
def calculateAverageColourOfContour(inputImageR, cnt):
	x,y,w,h = cv2.boundingRect(cnt) # offsets - with this you get 'mask'
	cv2.rectangle(inputImageR, (x,y), (x+w,y+h), (0,255,0), 2)
	#Average color (BGR):
	averageColour = cv2.mean(inputImageR[y:y+h,x:x+w])
	#np.array(averageColour).astype(np.uint8))
	#print("averageColour = ", averageColour)
	return averageColour
					

def main(inputimagefilename):
	#detectEllipsesTrialResize(inputimagefilename)
	detectEllipsesGaussianBlur(inputimagefilename)

if __name__ == "__main__":

	if len(sys.argv) < 2:
		print("Usage: python ATORpt_RFmainSA.py <input_image>")
		sys.exit(1)
	input_image_path = sys.argv[1]
	
	main(input_image_path)
