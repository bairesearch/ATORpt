"""ATORpt_RFapplyFilter.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORpt_RFmainFT.py

# Usage:
See ATORpt_RFmainFT.py

# Description:
ATORpt RF Filter - RF Filter transformations (pixel space)

"""

import torch as pt
import torch.nn.functional as F
import numpy as np
import copy
import cv2

from ATORpt_RFglobalDefs import *
import ATORpt_pta_image as pta_image
import ATORpt_RFpropertiesClass
import ATORpt_RFgenerateEllipse
import ATORpt_RFgenerateTri
import ATORpt_RFoperations


def calculateFilterApplicationResultThreshold(filterApplicationResult, minimumFilterRequirement, filterSize, isColourFilter, numberOfDimensions, RFtype):
	minimumFilterRequirementLocal = minimumFilterRequirement * calculateFilterPixels(filterSize, numberOfDimensions, RFtype)

	if(RFuseParallelProcessedCNN):
		pass
		#TODO: minimumFilterRequirementLocal REQUIRES CALIBRATION based on CNN operation
	else:
		# if(isColourFilter):
		# 	minimumFilterRequirementLocal = minimumFilterRequirementLocal*rgbNumChannels*rgbNumChannels  # CHECKTHIS  # not required as assume filter colours will be normalised to the maximum value of a single rgb channel?
		if not ATORpt_RFoperations.storeRFfiltersValuesAsFractions:
			minimumFilterRequirementLocal = minimumFilterRequirementLocal * (ATORpt_RFoperations.rgbMaxValue * ATORpt_RFoperations.rgbMaxValue)  # rgbMaxValue of both imageSegment and RFfilter

	print("minimumFilterRequirementLocal = ", minimumFilterRequirementLocal)
	print("pt.max(filterApplicationResult) = ", pt.max(filterApplicationResult))

	filterApplicationResultThreshold = filterApplicationResult > minimumFilterRequirementLocal
	return filterApplicationResultThreshold


def calculateFilterPixels(filterSize, numberOfDimensions, RFtype):
	if RFtype == RFtypeEllipse:
		return ATORpt_RFgenerateEllipse.calculateFilterPixels(filterSize, numberOfDimensions)
	elif RFtype == RFtypeTri:
		return ATORpt_RFgenerateTri.calculateFilterPixels(filterSize, numberOfDimensions)	#CHECKTHIS
	elif RFtype == RFtypeTemporaryPointFeatureKernel:
		return ATORpt_RFgenerateTri.calculateFilterPixels(filterSize, numberOfDimensions)	#CHECKTHIS

def normaliseRFfilter(RFfilter, RFproperties):
	# normalise ellipse respect to major/minor ellipticity axis orientation (WRT self)
	RFfilterNormalised = transformRFfilter(RFfilter, RFproperties)
	# RFfilterNormalised = RFfilter
	return RFfilterNormalised


def transformRFfilter(RFfilter, RFpropertiesParent):
	if RFpropertiesParent.numberOfDimensions == 2:
		centerCoordinates = [-RFpropertiesParent.centerCoordinates[0], -RFpropertiesParent.centerCoordinates[1]]
		#print("centerCoordinates = ", centerCoordinates)
		axesLength = 1.0 / RFpropertiesParent.axesLength[0]  # [1.0/RFpropertiesParent.axesLength[0], 1.0/RFpropertiesParent.axesLength[1]]
		angle = -RFpropertiesParent.angle
		RFfilterTransformed = transformRFfilter2D(RFfilter, centerCoordinates, axesLength, angle)
	elif RFpropertiesParent.numberOfDimensions == 3:
		print("error transformRFfilterWRTparent: RFpropertiesParent.numberOfDimensions == 3 not yet coded")
		quit()
	return RFfilterTransformed


def transformRFfilter2D(RFfilter, centerCoordinates, axesLength, angle):
	# CHECKTHIS: 2D code only;
	RFfilter = RFfilter.permute(2, 0, 1)	#ensure channels dim is first
	#RFfilterTransformed = pt.unsqueeze(RFfilter, 0)  # add batch dim
	RFfilterTransformed = RFfilter
	angleRadians = ATORpt_RFoperations.convertDegreesToRadians(angle)
	#print("RFfilterTransformed.shape = ", RFfilterTransformed.shape)
	RFfilterTransformed = pta_image.rotate(RFfilterTransformed, angleRadians, fillValue=RFfilterImageTransformFillValue)
	centerCoordinatesList = [float(x) for x in list(centerCoordinates)]
	RFfilterTransformed = pta_image.translate(RFfilterTransformed, centerCoordinatesList, fillValue=RFfilterImageTransformFillValue)
	# print("axesLength = ", axesLength)
	#print("RFfilterTransformed.shape = ", RFfilterTransformed.shape)
	RFfilterTransformed = pta_image.scale(RFfilterTransformed, axesLength, fillValue=RFfilterImageTransformFillValue)
	#print("RFfilterTransformed.shape = ", RFfilterTransformed.shape)
	RFfilterTransformed = pt.squeeze(RFfilterTransformed)
	RFfilter = RFfilter.permute(1, 2, 0)	#ensure channels dim is last
	return RFfilterTransformed

def rotateRFfilter(RFfilter, RFproperties):
	return rotateRFfilter(-RFproperties.angle)


def rotateRFfilter(RFfilter, angle):
	RFfilter = pt.unsqueeze(RFfilter, 0)  # add extra dimension for num_images
	return RFfilterNormalised


def getFilterDimensions(resolutionProperties):
	return ATORpt_RFpropertiesClass.getFilterDimensions(resolutionProperties)


# CHECKTHIS: upgrade code to support ATORpt_RFgenerateTri
def allFilterCoordinatesWithinImage(centerCoordinates, filterRadius, imageSize):
	imageSegmentStart = (centerCoordinates[0] - filterRadius, centerCoordinates[1] - filterRadius)
	imageSegmentEnd = (centerCoordinates[0] + filterRadius, centerCoordinates[1] + filterRadius)
	if (imageSegmentStart[0] >= 0 and imageSegmentStart[1] >= 0 and imageSegmentEnd[0] < imageSize[0] and
			imageSegmentEnd[1] < imageSize[1]):
		result = True
	else:
		result = False
		# create artificial image segment (will be discarded during image filter application)
		imageSegmentStart = (0, 0)
		imageSegmentEnd = (filterRadius * 2, filterRadius * 2)
	return result, imageSegmentStart, imageSegmentEnd


		
def crop_ellipse_area(image, ellipse_props, padding_ratio=0.5):
	"""
	Crop the region around the ellipse (with extra padding).
	Return the cropped patch and the top-left corner of this patch in the original image.
	"""

	(xc, yc) = ellipse_props.centerCoordinates
	(ax1, ax2) = ellipse_props.axesLength
	angle = ellipse_props.angle
	
	# If you're sure ax1 >= ax2, then ax1 is major axis length, ax2 is minor axis.
	# Otherwise, you might need to check and swap if needed:
	# if ax2 > ax1: 
	#	 ax1, ax2 = ax2, ax1  # swap so ax1 is always major, for clarity

	# Create a RotatedRect (center, (width, height), angle)
	# NOTE: OpenCV ellipse angle is measured from the X-axis in degrees.
	# For cv2.ellipse or RotatedRect, the "size" is (axis_width, axis_height)
	rot_rect = ((xc, yc), (ax1, ax2), angle)
	
	# Get the 4 corners of this rotated rectangle
	box_points = cv2.boxPoints(rot_rect)  # returns 4 points
	box_points = np.int0(box_points)	  # integer coords

	# Compute the bounding rect of those points
	x, y, w, h = cv2.boundingRect(box_points)

	# Expand that bounding box by `padding_ratio` on each side
	#  - "eg 50% padding" means we add 0.5 * w to the left/right and 0.5 * h to the top/bottom (split in half each side)
	pad_w = int(w * padding_ratio)
	pad_h = int(h * padding_ratio)

	x1 = x - pad_w
	y1 = y - pad_h
	x2 = x + w + pad_w
	y2 = y + h + pad_h

	# Clip to image boundaries
	H, W = image.shape[:2]
	x1_clip = max(0, x1)
	y1_clip = max(0, y1)
	x2_clip = min(W, x2)
	y2_clip = min(H, y2)

	# Crop from the original image
	cropped = image[y1_clip:y2_clip, x1_clip:x2_clip].copy()

	# We now place this cropped region onto a black canvas of size (y2-y1, x2-x1)
	out_h = (y2 - y1)
	out_w = (x2 - x1)
	patch = np.zeros((out_h, out_w, 3), dtype=image.dtype)

	# The region within patch that corresponds to the actual image data
	offset_x = x1_clip - x1  # how much we had to clip left
	offset_y = y1_clip - y1  # how much we had to clip top

	patch[offset_y:offset_y+cropped.shape[0], offset_x:offset_x+cropped.shape[1]] = cropped

	# Return the patch, plus the top-left (x1,y1) so we know how to map back to original coords
	return patch, (x1, y1)


def transform_patch(patch, ellipse_props, patch_topleft):
	"""
	Rotate, translate, and scale the patch so that the ellipse becomes
	a 100x100 circle in a 200x200 output image with center at (100,100).
	
	Returns the 200x200 transformed image.
	"""
	(xc, yc) = ellipse_props.centerCoordinates
	(ax1, ax2) = ellipse_props.axesLength
	angle_deg = ellipse_props.angle
	
	# Depending on your data, confirm which is major vs minor
	majorAxis = ax1
	minorAxis = ax2
	# If not sure which is larger, do:
	# majorAxis = max(ax1, ax2)
	# minorAxis = min(ax1, ax2)
	# but also note if the angle corresponds to the first or second axis in OpenCV's ellipse definition.

	#print("angle_deg = ", angle_deg)
	# Convert angle to radians for math
	angle_rad = np.deg2rad(-angle_deg)	# negative for the rotation we want

	# Patch coordinates of the ellipse center
	(x1, y1) = patch_topleft
	cx_patch = xc - x1
	cy_patch = yc - y1

	# Scale factors to turn ellipse into 100x100 circle
	Sx = 100.0 / minorAxis
	Sy = 100.0 / majorAxis

	cosA = np.cos(angle_rad)
	sinA = np.sin(angle_rad)
	
	# Build the 2x3 affine matrix M for warpAffine
	# [ a11  a12  b1 ]
	# [ a21  a22  b2 ]
	
	# After the shift-to-origin, rotate, scale, then translate to (100,100).
	# x_out = Sx*( (x - cx_patch)*cosA - (y - cy_patch)*sinA ) + 100
	# y_out = Sy*( (x - cx_patch)*sinA + (y - cy_patch)*cosA ) + 100

	a11 =  Sx * cosA
	a12 = -Sx * sinA
	a21 =  Sy * sinA
	a22 =  Sy * cosA

	b1 = 100.0 - (a11*cx_patch + a12*cy_patch)
	b2 = 100.0 - (a21*cx_patch + a22*cy_patch)

	M = np.array([[a11, a12, b1],
				  [a21, a22, b2]], dtype=np.float32)

	# Warp into 200x200 with black padding
	out_size = (200, 200)  # width, height
	warped = cv2.warpAffine(
		patch,
		M,
		out_size,
		flags=cv2.INTER_LINEAR,
		borderMode=cv2.BORDER_CONSTANT,
		borderValue=(0, 0, 0)
	)
	
	return warped


