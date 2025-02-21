"""ATORpt_PTfeatures.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt PT features - various third party feature detectors

"""

import torch as pt
import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import json

from ATORpt_PTglobalDefs import *

def featureDetection(image, zoomIndex):
	zoom = getZoomValue(zoomIndex)
	imageFeatureCoordinates = pt.zeros([0, 2])
	if(useFeatureDetectionCorners):
		imageFeatureCoordinates = pt.cat((imageFeatureCoordinates, featureDetectionCornerOpenCVHarris(image)), dim=0)
		#imageFeatureCoordinates = pt.cat((imageFeatureCoordinates, featureDetectionCornerOpenCVShiTomasi(image)), dim=0)
	if(useFeatureDetectionCentroids):
		imageFeatureCoordinates = pt.cat((imageFeatureCoordinates, featureDetectionCentroidFBSegmentAnything(image)), dim=0)
	imageFeatureCoordinates = imageFeatureCoordinates*zoom	#ensure feature coordinates are defined with respect to original image coordinates
	return imageFeatureCoordinates

def getZoomValue(zoomIndex):
	#sync with ATORmethodClass::createOrAddPointsToFeaturesList
	zoom = int(pow(2, zoomIndex))		#1, 2, 4, 8, 16 etc
	return zoom

def featureDetectionCornerOpenCVHarris(image):
	if(debugVerbose):
		print("featureDetectionCornerOpenCVHarris:")
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image_float32 = np.float32(image)
	dst = cv2.cornerHarris(image_float32, 2, 3, 0.04)	#cv2.cuda
	#dst = dst.download()
	
	cornerFeatureList = extractFeatureCoordsFromFeatureMapSubpixel(dst, image)
	return cornerFeatureList

def featureDetectionCornerOpenCVShiTomasi(image):
	if(debugVerbose):
		print("featureDetectionCornerOpenCVShiTomasi:")
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image_float32 = np.float32(image)
	dst = cv2.cornerMinEigenVal(image_float32, 3, 3, 3)	#cv2.cuda
	#dst = dst.download()
	
	cornerFeatureList = extractFeatureCoordsFromFeatureMapSubpixel(dst, image)
	if(debugVerbose):
		print("cornerFeatureList = ", cornerFeatureList)
	return cornerFeatureList

def extractFeatureCoordsFromFeatureMapSubpixel(dst, image):
	dst = cv2.dilate(dst,None)
	ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
	dst = np.uint8(dst)
	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
	corners = cv2.cornerSubPix(image,np.float32(centroids),(5,5),(-1,-1),criteria)	#respects xAxisFeatureMap/yAxisFeatureMap
	
	cornerFeatures = pt.tensor(corners, dtype=pt.float32) 	#store features as x, y - see xAxisFeatureMap/yAxisFeatureMap
	if(debugVerbose):
		print("cornerFeatures len = ", cornerFeatures.shape[0])
	
	if(debugFeatureDetection):
		printFeaturesArraySubpixel(corners, image)
	
	return cornerFeatures
		
def printFeaturesArraySubpixel(corners, image):
	corners = np.int0(corners)
	image[corners[:,yAxisFeatureMap], corners[:,xAxisFeatureMap]] = 255
	cv2.imshow('Corner Detection', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def extractFeatureCoordsFromFeatureMap(dst, image):
	dilation_size = 4
	kernel = np.ones((dilation_size, dilation_size), np.uint8)  # Create a kernel for dilation
	dst = cv2.dilate(dst, kernel)  # Dilate to find local maxima
	corner_mask = dst == cv2.erode(dst, kernel)  # Erode to find local minima
	dst *= corner_mask
		
	threshold = 0.02 * dst.max()
	coords = np.argwhere(dst > threshold)
	
	#ATOR assumes y coordinates start from bottom and go to top (resolved via renderInvertedYaxisToDisplayOriginalImagesUpright: snapshotRenderCameraRotationZaxis = 180 instead):
	#imageHeight = image.shape[0]
	#coords[:, 0] = -coords[: , 0] + imageHeight
	
	cornerFeatureList = [(coord[xAxisImages], coord[yAxisImages]) for coord in coords]	#store features as x, y - see xAxisFeatureMap/yAxisFeatureMap
	if(debugVerbose):
		print("len cornerFeatureList = ", len(cornerFeatureList))
	
	if(debugFeatureDetection):
		printFeatureDetectionMap(dst, image, threshold)
	
	return cornerFeatureList

def printFeatureDetectionMap(dst, image, threshold):
	image[dst > threshold] = 255
	cv2.imshow('Corner Detection', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
def featureDetectionCentroidFBSegmentAnything(image):
	#image (np.ndarray): The image to generate masks for, in HWC uint8 format
	centroidFeatureList = []
	if(debugVerbose):
		print("featureDetectionCentroidFBSegmentAnything:")
	sam = sam_model_registry[segmentAnythingViTHSAMname](checkpoint=segmentAnythingViTHSAMpathName)	#default model type
	mask_generator = SamAutomaticMaskGenerator(sam)
	masks = mask_generator.generate(image)	#imageName
	for segmentationIndex, segmentationMask in enumerate(masks):
		centroid = calculateMaskCentroid(segmentationMask['segmentation'])
		centroidFeatureList.append(centroid)
	centroidFeatures = pt.tensor(centroidFeatureList, dtype=pt.float32)
	if(debugFeatureDetection):
		printFeatureListSubpixel(centroidFeatureList, image)
	if(debugVerbose):
		print("centroidFeatures len = ", centroidFeatures.shape[0])
	return centroidFeatures

def calculateMaskCentroid(mask):
	y_indices, x_indices = np.indices(mask.shape)
	masked_x_indices = x_indices[mask]
	masked_y_indices = y_indices[mask]
	centroid_x = np.mean(masked_x_indices)
	centroid_y = np.mean(masked_y_indices)
	centroid = (centroid_x, centroid_y)	#store features as x, y - see xAxisFeatureMap/yAxisFeatureMap
	return centroid

def printFeatureListSubpixel(cornerFeatureList, image):
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	for corner in cornerFeatureList:
		x = int(corner[xAxisFeatureMap])
		y = int(corner[yAxisFeatureMap])
		image[y, x] = 255
	cv2.imshow('Corner Detection', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
