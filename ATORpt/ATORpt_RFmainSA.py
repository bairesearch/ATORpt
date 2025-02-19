"""ATORpt_RFmainSA.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORpt_RFmainFT.py

# Usage:
source activate pytorch3d
python ATORpt_RFmainSA.py images/leaf1.png

# Description:
Perform ATOR receptive field (RF) ellipse detection using segment-anything (SA) library (hardware accelerated) rather than RF filters.

"""

import sys
import os
import cv2
import numpy as np
import torch as pt
import matplotlib.pyplot as plt

from ATORpt_RFglobalDefs import *
import ATORpt_RFellipsePropertiesClass
import ATORpt_RFoperations
import ATORpt_RFapplyFilter

from segment_anything import sam_model_registry, SamPredictor
segmentAnythingViTHSAMpathName = "../segmentAnythingViTHSAM/sam_vit_h_4b8939.pth"

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")


def main(image_path):
	# Read image
	image_rgb = read_image(image_path)

	resolutionIndexMax = numberOfResolutions
	if RFscaleImage:
		resolutionIndexMin = resolutionIndexFirst
	else:
		resolutionIndexMin = resolutionIndexMax
	
	ellipsePropertiesListAllRes = []
	for resolutionIndex in range(resolutionIndexMin, resolutionIndexMax+1):
		ellipsePropertiesList = detectRFs(image_rgb, resolutionIndex)
		ellipsePropertiesListAllRes = ellipsePropertiesListAllRes + ellipsePropertiesList
	
	for ellipse in ellipsePropertiesListAllRes:
		generateNormalisedImageSegment(ellipse, image_rgb)
		

def detectRFs(image_rgb, resolutionIndex):
	inputImageHeight, inputImageWidth, inputImageChannels = image_rgb.shape
	imageSizeBase = (inputImageWidth, inputImageHeight)
	
	resolutionProperties = ATORpt_RFoperations.RFresolutionProperties(resolutionIndex, resolutionIndexFirst, numberOfResolutions, imageSizeBase)
	#print("resolutionIndex = ", resolutionIndex)
	#print("resolutionProperties.resolutionFactor = ", resolutionProperties.resolutionFactor)
	#print("resolutionProperties.imageSize = ", resolutionProperties.imageSize)

	resizedImage = cv2.resize(image_rgb, resolutionProperties.imageSize, interpolation=cv2.INTER_LINEAR)

	# Detect segments
	features = detect_segments(resizedImage, sam_checkpoint=segmentAnythingViTHSAMpathName)

	# Detect ellipses
	ellipsePropertiesList = detect_ellipses(features, resolutionProperties)
	
	# Draw original image + outline
	draw_original_image_and_outline(resizedImage, features)
	
	return ellipsePropertiesList
	
def read_image(image_path):
	"""
	2. Read an image file into memory (BGR with OpenCV).
	   Convert to RGB for consistent processing/visualization.
	"""
	if not os.path.exists(image_path):
		raise FileNotFoundError(f"Image file not found: {image_path}")

	image_bgr = cv2.imread(image_path)
	if image_bgr is None:
		raise ValueError(f"Could not read image from: {image_path}")

	image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
	return image_rgb

def compute_contrast_map(image_rgb, method='laplacian', ksize=3):
	image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
	if method == 'laplacian':
		# Apply Laplacian filter to detect edges/contrast
		image_contrast = cv2.Laplacian(image_gray, cv2.CV_64F, ksize=ksize)
		image_contrast = np.abs(image_contrast)  # Take absolute value
	elif method == 'std':
		# Compute local standard deviation for contrast estimation
		blurred = cv2.GaussianBlur(image_gray, (ksize, ksize), 0)
		image_contrast = np.sqrt(cv2.GaussianBlur(image_gray**2, (ksize, ksize), 0) - blurred**2)
	else:
		raise ValueError("Invalid method. Choose 'laplacian' or 'std'.")
	# Normalize to 0-255
	image_contrast = cv2.normalize(image_contrast, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
	return image_contrast
	
def detect_segments(image_rgb, sam_checkpoint=None):

	height, width, _ = image_rgb.shape
	
	image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
	image_contrast = compute_contrast_map(image_rgb, method='laplacian')
	
	# use Segment Anything to get segment masks
	masks = []
	if sam_checkpoint is not None:
		sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
		sam.to(device)
		predictor = SamPredictor(sam)
		predictor.set_image(image_rgb)
		from segment_anything import SamAutomaticMaskGenerator
		mask_generator = SamAutomaticMaskGenerator(sam)
		masks = mask_generator.generate(image_rgb)
	else:
		print("detect_segments error: SAM checkpoint not provided.")
	
	segment_points = []
	edge_points = []
	centroid_points = []
	colour_points = []

	if len(masks) == 0:
		print("detect_segments error: no SAM masks found or checkpoint missing.")
	else:
		for m in masks:
			# m is a dict with keys 'segmentation', 'area', 'bbox', ...
			seg = m["segmentation"].astype(np.uint8)
		
			# Edge detection - find the boundary of the segment (contour)
			contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
			edge_points_segment = []
			for c in contours:	#should only be one contour per segment
				for pt in c:
					ix = pt[0][0]	#x
					iy = pt[0][1]	#y
					if ix < 0 or ix >= width or iy < 0 or iy >= height:
						continue
					contrast = float(image_contrast[iy, ix])  # simplistic contrast measure
					edge_points_segment.append((ix, iy, contrast))
			edge_points_segment_np = np.array(edge_points_segment, dtype=np.float32)
			edge_points.append(edge_points_segment_np)
			
			# Colour detection
			mask = seg.astype(bool)
			colour_segment = [image_rgb[:, :, c][mask].mean() for c in range(3)]  # R, G, B order
			colour_points.append(colour_segment)
			
			# Segment coordinates derivation
			segment_points_segment = np.argwhere(seg > 0)  # Returns (row, col) -> (y, x)
			segment_points_segment = segment_points_segment[:, [1, 0]]  # Swap columns to get (x, y)
			segment_points.append(segment_points_segment)
			
			# Centroid detection
			M = cv2.moments(seg, binaryImage=True)
			if M["m00"] != 0:
				cx = M["m10"] / M["m00"]
				cy = M["m01"] / M["m00"]
				# get contrast from the center pixel
				ix, iy = int(round(cx)), int(round(cy))
				if ix >= 0 and ix < width and iy >= 0 and iy < height:
					contrast = float(image_contrast[iy, ix])
					centroid_points.append((cx, cy, contrast))

	features = {
		"segment_points": segment_points,
		"edge_points": edge_points,
		"centroid_points": centroid_points,
		"colour_points": colour_points,
	}
	
	return features

def detect_ellipses(features, resolutionProperties):
	segment_points = features["segment_points"]
	colour_points = features["colour_points"]
	ellipsePropertiesList = []
	
	for segment_index, segment_points_segment in enumerate(segment_points):
		colour = colour_points[segment_index]
		if len(segment_points_segment) >= 5:

			ellipse = cv2.fitEllipse(segment_points_segment.astype(np.float32))
			centerCoordinates, axesLength, angle = rescaleEllipseCoordinates(ellipse, resolutionProperties)
			ellipseProperties = ATORpt_RFellipsePropertiesClass.EllipsePropertiesClass(centerCoordinates, axesLength, angle, colour)
			#inputImageRmod, ellipseFitError = ATORpt_RFellipsePropertiesClass.testEllipseApproximation(inputImageR, ellipseProperties)
			ellipsePropertiesList.append(ellipseProperties)

			if(debugVerbose):
				print("\nEllipse #", segment_index)
				(cx, cy) = centerCoordinates
				(majorAxis, minorAxis) = axesLength
				print(f"Center: ({cx:.2f}, {cy:.2f})")
				print(f"Major Axis: {majorAxis:.2f}")
				print(f"Minor Axis: {minorAxis:.2f}")
				print(f"Rotation Angle: {angle:.2f}")
				print("colour = ", colour)

	return ellipsePropertiesList

def rescaleEllipseCoordinates(ellipse, resolutionProperties):
	centerCoordinates, axesLength, angle = ellipse
	
	(cx, cy) = centerCoordinates
	(majorAxis, minorAxis) = axesLength
	cx = cx*resolutionProperties.resolutionFactor
	cy = cy*resolutionProperties.resolutionFactor
	majorAxis = majorAxis*resolutionProperties.resolutionFactor
	minorAxis = minorAxis*resolutionProperties.resolutionFactor
	centerCoordinates = (cx, cy)
	axesLength = (majorAxis, minorAxis)
	
	return centerCoordinates, axesLength, angle

def draw_original_image_and_outline(image_rgb, features):

	segment_points = features["segment_points"]
	edge_points = features["edge_points"]
	centroid_points = features["centroid_points"]
	centroid_points = np.array(centroid_points, dtype=np.float32)
	
	plt.figure("Step 5 - Original + Outline", figsize=(8, 6))
	plt.imshow(image_rgb)
	plt.title("Original Image + Mesh Outline")
	
	for segment_points_segment in segment_points:
		plt.scatter(segment_points_segment[:,0], segment_points_segment[:,1], c="blue", s=1)
	for edge_points_segment in edge_points:
		plt.scatter(edge_points_segment[:,0], edge_points_segment[:,1], c="green", s=2)
	plt.scatter(centroid_points[:,0], centroid_points[:,1], c="red", s=3)

	plt.axis("off")
	plt.show(block=True)  # block=False so we can continue
	
def draw_image(patch_rgb, name):	
	plt.figure(name, figsize=(8, 6))
	plt.imshow(patch_rgb)
	plt.title(name)
	plt.axis("off")
	plt.show(block=True)  # block=False so we can continue
	

def generateNormalisedImageSegment(ellipse, image_rgb):
	
	patch, patch_topleft = ATORpt_RFapplyFilter.crop_ellipse_area(image_rgb, ellipse, padding_ratio=1.0)
	draw_image(patch, "patch_orig")
	patch_transformed= ATORpt_RFapplyFilter.transform_patch(patch, ellipse, patch_topleft)
	draw_image(patch_transformed, "patch_transformed")	#200x200 pixels, with the transformed ellipse (now circle) occupying the centre 100x100 pixels

	'''
	segmentImage, patch_topleft = generateSegment(ellipse, image_rgb)
	RFproperties = ellipse
	RFproperties.numberOfDimensions = 2
	RFproperties.centerCoordinates = (0.0, 0.0)	#segment image is already centred
	RFfilter = pt.tensor(segmentImage, dtype=pt.float32, device=device)
	RFfilterTransformed = ATORpt_RFapplyFilter.normaliseRFfilter(RFfilter, RFproperties)
	'''
	
	return patch_transformed


if __name__ == "__main__":

	if len(sys.argv) < 2:
		print("Usage: python ATORpt_RFmainSA.py <input_image>")
		sys.exit(1)
	input_image_path = sys.argv[1]
	
	main(input_image_path)


def generateSegment(ellipse, image_rgb):
	inputImageHeight, inputImageWidth, inputImageChannels = image_rgb.shape

	segmentRadius = int(max(ellipse.axesLength[0], ellipse.axesLength[1])//2)	#take square region
	segmentCoordinates = (int(ellipse.centerCoordinates[0]), int(ellipse.centerCoordinates[1]))
	
	#ensure segment radius does not go outside of image (crop);
	if(segmentCoordinates[0]-segmentRadius < 0):
		segmentRadius = segmentCoordinates[0]
	if(segmentCoordinates[1]-segmentRadius < 1):
		segmentRadius = segmentCoordinates[1]
	if(segmentCoordinates[0]+segmentRadius > inputImageWidth):
		segmentRadius = inputImageWidth-segmentCoordinates[0]
	if(segmentCoordinates[1]+segmentRadius > inputImageHeight):
		segmentRadius = inputImageHeight-segmentCoordinates[1]
	
	#print("segmentRadius = ", segmentRadius)
	#print("segmentCoordinates = ", segmentCoordinates)
	
	patch_topleft = segmentCoordinates[0]-segmentRadius, segmentCoordinates[1]-segmentRadius
	segmentImage = image_rgb[segmentCoordinates[0]-segmentRadius:segmentCoordinates[0]+segmentRadius, segmentCoordinates[1]-segmentRadius:segmentCoordinates[1]+segmentRadius]

	return segmentImage, patch_topleft
