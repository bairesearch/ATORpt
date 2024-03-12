"""ATORpt_PTfeatureDetector.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt PT feature detector - variod third party feature detectors

"""

import torch as pt
import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import json

from ATORpt_globalDefs import *

def featureDetection(image, zoomIndex):
	zoom = getZoomValue(zoomIndex)
	imageFeatureCoordinatesList = []
	if(useFeatureDetectionCorners):
		imageFeatureCoordinatesList = imageFeatureCoordinatesList + featureDetectionCornerOpenCVHarris(image)
		#imageFeatureCoordinatesList = imageFeatureCoordinatesList + featureDetectionCornerOpenCVShiTomasi(image)
	if(useFeatureDetectionCentroids):
		imageFeatureCoordinatesList = imageFeatureCoordinatesList + featureDetectionCentroidFBSegmentAnything(image)
	imageFeatureCoordinates = pt.tensor(imageFeatureCoordinatesList, dtype=pt.float32)	#double->float conversion required for featureDetectionCentroidFBSegmentAnything:calculateMaskCentroid
	#print("imageFeatureCoordinates.shape = ", imageFeatureCoordinates.shape)
	imageFeatureCoordinates = imageFeatureCoordinates*zoom	#ensure feature coordinates are defined with respect to original image coordinates
	return imageFeatureCoordinates

def getZoomValue(zoomIndex):
	#sync with ATORmethodClass::createOrAddPointsToFeaturesList
	zoom = int(pow(2, zoomIndex))		#1, 2, 4, 8, 16 etc
	return zoom

def featureDetectionCornerOpenCVHarris(image):
	print("featureDetectionCornerOpenCVHarris:")
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image_float32 = np.float32(image)
	dst = cv2.cornerHarris(image_float32, 2, 3, 0.04)	#cv2.cuda
	#dst = dst.download()
	
	cornerFeatureList = extractFeatureCoordsFromFeatureMap(dst, image)
	return cornerFeatureList

def featureDetectionCornerOpenCVShiTomasi(image):
	print("featureDetectionCornerOpenCVShiTomasi:")
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image_float32 = np.float32(image)
	dst = cv2.cornerMinEigenVal(image_float32, 3, 3, 3)	#cv2.cuda
	#dst = dst.download()
	
	cornerFeatureList = extractFeatureCoordsFromFeatureMap(dst, image)
	print("cornerFeatureList = ", cornerFeatureList)
	return cornerFeatureList

def extractFeatureCoordsFromFeatureMap(dst, image):
	dilation_size = 4
	kernel = np.ones((dilation_size, dilation_size), np.uint8)  # Create a kernel for dilation
	dst = cv2.dilate(dst, kernel)  # Dilate to find local maxima
	corner_mask = dst == cv2.erode(dst, kernel)  # Erode to find local minima
	dst *= corner_mask
		
	threshold = 0.02 * dst.max()
	coords = np.argwhere(dst > threshold)
	cornerFeatureList = [(coord[1], coord[0]) for coord in coords]	#store features as x, y - see xAxisFeatureMap/yAxisFeatureMap

	#print("cornerFeatureList = ", cornerFeatureList)
	print("len cornerFeatureList = ", len(cornerFeatureList))
	
	#printFeatureDetectionMap(dst, image, threshold)	#debug
	
	return cornerFeatureList

def printFeatureDetectionMap(dst, image, threshold):
	image[dst > threshold] = 255
	cv2.imshow('Corner Detection', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
def featureDetectionCentroidFBSegmentAnything(image):
	#image (np.ndarray): The image to generate masks for, in HWC uint8 format
	centroidFeatureList = []
	print("featureDetectionCentroidFBSegmentAnything:")
	sam = sam_model_registry["vit_h"](checkpoint=segmentAnythingViTHSAMpathName)	#default model type
	mask_generator = SamAutomaticMaskGenerator(sam)
	masks = mask_generator.generate(image)	#imageName
	for segmentationIndex, segmentationMask in enumerate(masks):
		centroid = calculateMaskCentroid(segmentationMask['segmentation'])
		centroidFeatureList.append(centroid)
	#print("masks: ", masks[0]['segmentation'])
	#print("centroidFeatureList = ", centroidFeatureList)
	print("len centroidFeatureList = ", len(centroidFeatureList))

	return centroidFeatureList

def calculateMaskCentroid(mask):
	# Calculate moments
	m00 = np.sum(mask)
	m10 = np.sum(np.arange(mask.shape[0])[:, None] * mask)
	m01 = np.sum(np.arange(mask.shape[1])[None, :] * mask)

	# Calculate centroid
	if m00 == 0:
		centroid_x, centroid_y = 0, 0
	else:
		centroid_x = m10 / m00
		centroid_y = m01 / m00

	centroid = (centroid_x, centroid_y)	#store features as x, y - see xAxisFeatureMap/yAxisFeatureMap
	return centroid
